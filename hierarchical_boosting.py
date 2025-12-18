"""
Universal Hierarchical Probability Boosting System
This module provides a flexible framework for boosting channel probabilities
based on hierarchical relationships and mutual exclusivity constraints.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class RelationType(Enum):
    """Types of relationships between markers"""
    MUTUALLY_EXCLUSIVE = "mutually_exclusive"
    SUBTYPE = "subtype"
    COEXPRESSION = "coexpression"
    INHIBITORY = "inhibitory"


@dataclass
class MarkerRelationship:
    """Defines a relationship between markers"""
    parent: str
    children: List[str]
    relation_type: RelationType
    boost_factor: float = 1.0
    ambiguity_threshold: float = 0.65
    

@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical boosting"""
    markers: List[str]
    relationships: List[MarkerRelationship]
    temperature: float = 0.3
    ambiguity_steepness: float = 5.0
    unknown_weight: float = 0.618
    include_unknown: bool = True
    

class HierarchicalProbabilityBooster:
    """
    Universal hierarchical probability booster that can handle
    multiple channels and complex relationships
    """
    
    def __init__(self, config: HierarchicalConfig):
        self.config = config
        self.marker_to_idx = {marker: i for i, marker in enumerate(config.markers)}
        
    def calculate_softmax(self, scores: np.ndarray, temperature: float = None) -> np.ndarray:
        """
        Calculate softmax probabilities with temperature scaling
        
        Parameters:
        scores: Raw scores for each channel
        temperature: Temperature parameter for softmax (lower = sharper)
        
        Returns:
        Softmax probabilities
        """
        if temperature is None:
            temperature = self.config.temperature
            
        # Numerical stability
        scores_scaled = scores / temperature
        scores_scaled = scores_scaled - np.max(scores_scaled, axis=1, keepdims=True)
        
        exp_scores = np.exp(scores_scaled)
        softmax_probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return softmax_probs
    
    def calculate_ambiguity(self, prob1: np.ndarray, prob2: np.ndarray) -> np.ndarray:
        """
        Calculate ambiguity between two probability distributions
        
        Parameters:
        prob1, prob2: Probability arrays
        
        Returns:
        Ambiguity scores
        """
        diff = np.abs(prob1 - prob2)
        ambiguity = np.exp(-self.config.ambiguity_steepness * diff)
        return ambiguity
    
    def apply_hierarchical_boost(self, 
                                  probs: np.ndarray, 
                                  relationship: MarkerRelationship) -> np.ndarray:
        """
        Apply hierarchical boosting based on marker relationships
        
        Parameters:
        probs: Current probability matrix
        relationship: Marker relationship definition
        
        Returns:
        Boosted probability matrix
        """
        parent_idx = self.marker_to_idx.get(relationship.parent)
        if parent_idx is None:
            return probs
            
        child_indices = [self.marker_to_idx[child] for child in relationship.children 
                        if child in self.marker_to_idx]
        
        if not child_indices:
            return probs
            
        boosted_probs = probs.copy()
        
        if relationship.relation_type == RelationType.SUBTYPE:
            # Boost parent based on children (e.g., CD4/CD8 boost CD3)
            helper = np.max(probs[:, child_indices], axis=1)
            
            # Find competing markers (mutually exclusive with parent)
            competing_markers = self._find_competing_markers(relationship.parent)
            
            if competing_markers:
                for competitor in competing_markers:
                    comp_idx = self.marker_to_idx.get(competitor)
                    if comp_idx is not None:
                        ambiguity = self.calculate_ambiguity(
                            probs[:, parent_idx], 
                            probs[:, comp_idx]
                        )
                        
                        # Apply boost only in ambiguous cases
                        mask = (ambiguity >= relationship.ambiguity_threshold)
                        
                        boost = ambiguity * helper * relationship.boost_factor
                        
                        boosted_probs[mask, parent_idx] = probs[mask, parent_idx] + \
                            (1 - probs[mask, parent_idx]) * boost[mask]
                            
        elif relationship.relation_type == RelationType.MUTUALLY_EXCLUSIVE:
            # Handle mutual exclusivity
            pass  # Already handled implicitly through softmax
            
        elif relationship.relation_type == RelationType.COEXPRESSION:
            # Boost both markers if they tend to co-occur
            for child_idx in child_indices:
                correlation = probs[:, parent_idx] * probs[:, child_idx]
                boost = correlation * relationship.boost_factor
                
                boosted_probs[:, parent_idx] += boost
                boosted_probs[:, child_idx] += boost
                
        elif relationship.relation_type == RelationType.INHIBITORY:
            # Reduce probability if inhibitory marker is high
            for child_idx in child_indices:
                inhibition = probs[:, child_idx] * relationship.boost_factor
                boosted_probs[:, parent_idx] = probs[:, parent_idx] * (1 - inhibition)
        
        return boosted_probs
    
    
    def _find_competing_markers(self, marker: str) -> List[str]:
        """Find markers that are mutually exclusive with the given marker"""
        competitors = []
        for rel in self.config.relationships:
            if rel.relation_type == RelationType.MUTUALLY_EXCLUSIVE:
                if marker == rel.parent:
                    competitors.extend(rel.children)
                elif marker in rel.children:
                    competitors.append(rel.parent)
                    competitors.extend([c for c in rel.children if c != marker])
        return list(set(competitors))
    
    def _global_ambiguity_mask(self, probs: np.ndarray, threshold: float) -> np.ndarray:
        """
        Create a boolean mask for ambiguous samples based on top-2 softmax probabilities.
        Ambiguity score = exp(-steepness * |p_top - p_second|).
        """
        # top-2 per row
        part_sorted = np.partition(probs, -2, axis=1)
        p_second = part_sorted[:, -2]
        p_top = part_sorted[:, -1]
        # reuse the same ambiguity transform
        # (needs arrays; shape them to 1D)
        ambiguity = self.calculate_ambiguity(p_top, p_second)
        return ambiguity >= threshold
    
    def apply_hierarchical_boost(
        self,
        probs: np.ndarray,
        relationship: MarkerRelationship
    ) -> np.ndarray:
        """
        Apply hierarchical boosting ONLY on ambiguous samples.
        """
        parent_idx = self.marker_to_idx.get(relationship.parent)
        if parent_idx is None:
            return probs

        child_indices = [self.marker_to_idx[child] for child in relationship.children
                         if child in self.marker_to_idx]
        if not child_indices:
            return probs

        boosted_probs = probs.copy()

        # --- NEW: global ambiguity gating (top-2) ---
        global_mask = self._global_ambiguity_mask(
            probs, relationship.ambiguity_threshold
        )
        if not np.any(global_mask):
            return boosted_probs  # nothing to do if no ambiguous rows

        if relationship.relation_type == RelationType.SUBTYPE:
            # Boost parent based on children (e.g., CD4/CD8 -> CD3), but only when:
            # (a) the row is globally ambiguous, AND
            # (b) parent is ambiguous against a competitor (as you originally did).
            helper = np.max(probs[:, child_indices], axis=1)

            # Find competing markers (mutually exclusive with parent)
            competing_markers = self._find_competing_markers(relationship.parent)

            if competing_markers:
                for competitor in competing_markers:
                    comp_idx = self.marker_to_idx.get(competitor)
                    if comp_idx is None:
                        continue

                    # parent-vs-competitor ambiguity
                    pair_ambiguity = self.calculate_ambiguity(
                        probs[:, parent_idx],
                        probs[:, comp_idx]
                    )
                    pair_mask = (pair_ambiguity >= relationship.ambiguity_threshold)

                    # final mask = globally ambiguous AND pair-ambiguous
                    mask = global_mask & pair_mask
                    if not np.any(mask):
                        continue

                    boost = pair_ambiguity * helper * relationship.boost_factor
                    boosted_probs[mask, parent_idx] = probs[mask, parent_idx] + \
                        (1 - probs[mask, parent_idx]) * boost[mask]

        elif relationship.relation_type == RelationType.MUTUALLY_EXCLUSIVE:
            # Softmax already imposes competition; nothing to do.
            pass

        elif relationship.relation_type == RelationType.COEXPRESSION:
            # Apply only on globally ambiguous rows
            for child_idx in child_indices:
                correlation = probs[:, parent_idx] * probs[:, child_idx]
                boost = correlation * relationship.boost_factor
                boosted_probs[global_mask, parent_idx] += boost[global_mask]
                boosted_probs[global_mask, child_idx] += boost[global_mask]

        elif relationship.relation_type == RelationType.INHIBITORY:
            # Apply only on globally ambiguous rows
            for child_idx in child_indices:
                inhibition = probs[:, child_idx] * relationship.boost_factor
                # multiplicative downscale for the masked rows
                scaled = probs[:, parent_idx] * (1 - inhibition)
                boosted_probs[global_mask, parent_idx] = scaled[global_mask]

        return boosted_probs
    
    def process(self, df: pd.DataFrame, 
                prob_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Process DataFrame with hierarchical boosting
        
        Parameters:
        df: Input DataFrame with probability columns
        prob_columns: List of column names to process (default: use config markers)
        
        Returns:
        DataFrame with boosted probabilities and final predictions
        """
        if prob_columns is None:
            prob_columns = [f"{marker}_prob" for marker in self.config.markers]
        
        # Extract probability matrix
        print('columns', prob_columns)
        prob_matrix = df[prob_columns].values
        
        # Add unknown channel if configured
        if self.config.include_unknown:
            unknown_score = np.prod(1 - prob_matrix, axis=1, keepdims=True)
            unknown_score *= self.config.unknown_weight
            prob_matrix = np.hstack([prob_matrix, unknown_score])
            extended_markers = self.config.markers + ['unknown']
        else:
            extended_markers = self.config.markers
        
        # Apply softmax normalization
        softmax_probs = self.calculate_softmax(prob_matrix)
        
        # Apply hierarchical boosting for each relationship
        for relationship in self.config.relationships:
            softmax_probs = self.apply_hierarchical_boost(softmax_probs, relationship)

        # Renormalize after boosting
        softmax_probs = softmax_probs / np.sum(softmax_probs, axis=1, keepdims=True)
        
        # Add results to DataFrame
        result_df = df.copy()
        
        # Add final probabilities
        for i, marker in enumerate(extended_markers):
            result_df[f"{marker}_final_prob"] = softmax_probs[:, i]
        
        # Add predicted channel
        result_df['predicted_channel'] = [extended_markers[i] 
                                         for i in np.argmax(softmax_probs, axis=1)]
        result_df['prediction_confidence'] = np.max(softmax_probs, axis=1)
        
        # Calculate ambiguity scores for key pairs
        for rel in self.config.relationships:
            if rel.relation_type == RelationType.MUTUALLY_EXCLUSIVE:
                parent_idx = self.marker_to_idx.get(rel.parent)
                for child in rel.children:
                    child_idx = self.marker_to_idx.get(child)
                    if parent_idx is not None and child_idx is not None:
                        ambiguity = self.calculate_ambiguity(
                            softmax_probs[:, parent_idx],
                            softmax_probs[:, child_idx]
                        )
                        result_df[f"ambiguity_{rel.parent}_{child}"] = ambiguity
        
        return result_df


# Example configurations moved to examples/test files
# The core module should remain generic without specific marker names
