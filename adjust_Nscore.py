import json
import pandas as pd
from agenttools import compute_marker_probabilities


def process_df(df,phenotype_filepath):
    with open(phenotype_filepath) as f:
        phenotype_data = json.load(f)
    for lev,tree in phenotype_data.items():
        # agent get the three markers variables
        primary_markers = tree['primary_markers']
        helper_markers = tree ['helper_markers']
        target_help_markers = tree['target_help_markers']
        
        compute_marker_probabilities(
        df,
        primary_markers,
        helper_markers,
        target_help_markers,)
        
    return df
    