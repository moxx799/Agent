import numpy as np

def compute_marker_probabilities(
    df,
    primary_markers,
    helper_markers,
    target_help_markers,
    T=0.3,
    k=4.0,
    unknown_weight=0.618,
    ambiguity_threshold=0.65,
    unknown_max_prob=0.5,
    ambiguity_prefix="ambiguity"
):
    """
    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns for primary_markers and all helper markers.
    primary_markers : list of str
        Marker columns used as primary classes (e.g. ['CD3', 'CD20']).
    helper_markers : list
        Either:
          - list of str (one helper per target), or
          - list of list/tuple/set of str (multiple helpers per target).
        Must be the same length as target_help_markers.
        Example:
            helper_markers = [
                ['CD4', 'CD8'],   # helpers for 'CD3'
                ['CD56']          # helpers for 'CD16'
            ]
    target_help_markers : list of str
        Subset of primary_markers that receive helper-based boosting.
        Paired index-wise with helper_markers.
    T : float
        Temperature for softmax.
    k : float
        Controls sharpness in ambiguity computation.
    unknown_weight : float
        Weight applied to unknown score.
    ambiguity_threshold : float
        Minimum ambiguity to apply helper boosting.
    unknown_max_prob : float
        Maximum allowed unknown softmax prob to still apply helper boosting.
    ambiguity_prefix : str
        Prefix for ambiguity column names.

    Returns
    -------
    df : pandas.DataFrame
        Modified in-place, but also returned for convenience.
    """

    # ---- Normalize helper_markers to list-of-lists ----
    if len(helper_markers) != len(target_help_markers):
        raise ValueError(
            "helper_markers and target_help_markers must have the same length "
            f"(got {len(helper_markers)} and {len(target_help_markers)})."
        )

    normalized_helper_groups = []
    for hm in helper_markers:
        # If it's a single string, wrap into a list
        if isinstance(hm, str):
            normalized_helper_groups.append([hm])
        # If it's any iterable of strings (list/tuple/set), convert to list
        elif isinstance(hm, (list, tuple, set)):
            normalized_helper_groups.append(list(hm))
        else:
            raise TypeError(
                "Each element of helper_markers must be either a string or a list/tuple/set of strings. "
                f"Got type {type(hm)}."
            )

    # Map marker -> index
    channel_columns = list(primary_markers)
    idx_map = {m: i for i, m in enumerate(channel_columns)}

    # Unknown score from product of (1 - primary_markers)
    primary_values = df[channel_columns].values
    unknown_score = np.prod(1 - primary_values, axis=1, keepdims=True) * unknown_weight

    # Softmax over primary + unknown
    combined_scores = np.hstack([primary_values, unknown_score])
    extended_channels = channel_columns + ['unknown']

    logits = combined_scores / T
    logits = logits - np.max(logits, axis=1, keepdims=True)  # numerical stability
    softmax_probs = np.exp(logits)
    softmax_probs = softmax_probs / np.sum(softmax_probs, axis=1, keepdims=True)

    unknown_idx = len(channel_columns)  # index of 'unknown'

    # ---- Ambiguity and boosting (first two primary markers define the ambiguity pair) ----
    if len(primary_markers) >= 2:
        a = primary_markers[0]
        b = primary_markers[1]
        a_idx = idx_map[a]
        b_idx = idx_map[b]

        # ambiguity between the first two primary markers
        diff = np.abs(softmax_probs[:, a_idx] - softmax_probs[:, b_idx])
        ambiguity = np.exp(-k * diff)

        # store initial ambiguity
        df[f'{ambiguity_prefix}_{a}_{b}_1'] = ambiguity

        # Masks: ambiguous AND not too unknown
        ambiguity_mask = ambiguity >= ambiguity_threshold
        unknown_mask = softmax_probs[:, unknown_idx] < unknown_max_prob
        process_mask = ambiguity_mask & unknown_mask

        # ---- helper-based boosting for each (helper_group, target) pair ----
        for helper_group, target_marker in zip(normalized_helper_groups, target_help_markers):
            # skip if target is not a primary channel
            if target_marker not in idx_map:
                continue

            # filter helper markers to those that actually exist in df
            valid_helpers = [h for h in helper_group if h in df.columns]
            if not valid_helpers:
                continue  # nothing to use

            m_idx = idx_map[target_marker]

            # helper signal: max over this group's helpers
            helper_vals = df[valid_helpers].max(axis=1).values
            boost = helper_vals * ambiguity  # scaled by ambiguity

            # apply boost only where process_mask is True
            boosted = softmax_probs[:, m_idx] + (1 - softmax_probs[:, m_idx]) * boost
            effective = np.where(process_mask, boosted, softmax_probs[:, m_idx])

            softmax_probs[:, m_idx] = effective

        # Renormalize only affected rows
        if np.any(process_mask):
            softmax_probs[process_mask] /= (
                np.sum(softmax_probs[process_mask], axis=1, keepdims=True) + 1e-8
            )

        # New ambiguity after boosting for the first two markers
        new_diff = np.abs(
            softmax_probs[:, a_idx] - softmax_probs[:, b_idx]
        )
        new_ambiguity = np.exp(-k * new_diff)
        df[f'{ambiguity_prefix}_{a}_{b}'] = new_ambiguity

    # ---- Write back results to df ----
    df['positive_channel'] = [
        extended_channels[i] for i in np.argmax(softmax_probs, axis=1)
    ]
    df['positive_prob'] = np.max(softmax_probs, axis=1)

    for i, col in enumerate(extended_channels):
        df[f'{col}_prob'] = softmax_probs[:, i]

    return df
