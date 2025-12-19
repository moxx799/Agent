import json
import pandas as pd
from agenttools import compute_marker_probabilities
from agents.get_phenotye_res import get_response
from agents.get_phenotype_hier import DictionaryAgent

def extract_markers_from_tree(tree):
    """
    Extract primary_markers, helper_markers, and target_help_markers from tree structure.
    
    Logic:
    - Primary markers: direct children of the top parent
    - For each primary marker that has children, those children are helpers
    - Target help markers: primary markers that have helper children
    """
    relationships = tree.get('relationships', [])
    
    # Find the top-level relationship
    top_relationship = None
    for rel in relationships:
        if 'Top_parent' in rel:
            top_relationship = rel
            break
    
    if not top_relationship:
        return [], [], []
    
    # Primary markers are the direct children of top parent
    primary_markers = top_relationship.get('children', [])
    
    # Build helper mapping: find which primary markers have children
    helper_markers = []
    target_help_markers = []
    
    for rel in relationships:
        if 'parent' in rel and rel['parent'] in primary_markers:
            parent = rel['parent']
            children = rel.get('children', [])
            if children:
                target_help_markers.append(parent)
                helper_markers.append(children)
    
    return primary_markers, helper_markers, target_help_markers

def process_nimbus(df, phenotype_filepath, **kwargs):
    """
    Process dataframe through hierarchical marker probability computation.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing marker intensity columns
    phenotype_filepath : str
        Path to JSON file with hierarchical marker structure
    **kwargs : dict
        Additional parameters to pass to compute_marker_probabilities
        (e.g., T, k, unknown_weight, ambiguity_threshold, etc.)
    
    Returns
    -------
    df : pandas.DataFrame
        Modified dataframe with probability columns added
    """
    with open(phenotype_filepath) as f:
        phenotype_data = json.load(f)
    
    for level_name, tree in phenotype_data.items():
        print(f"\nProcessing {level_name}...")
        
        # Extract markers from tree structure
        primary_markers, helper_markers, target_help_markers = extract_markers_from_tree(tree)
        
        if not primary_markers:
            print(f"  No primary markers found in {level_name}, skipping...")
            continue
        
        print(f"  Primary markers: {primary_markers}")
        print(f"  Target markers: {target_help_markers}")
        print(f"  Helper markers: {helper_markers}")
        
        # Check if required columns exist in dataframe
        missing_cols = [m for m in primary_markers if m not in df.columns]
        if missing_cols:
            print(f"  Warning: Missing columns {missing_cols}, skipping {level_name}")
            continue
        
        # Compute probabilities for this level
        compute_marker_probabilities(
            df,
            primary_markers,
            helper_markers,
            target_help_markers,
            **kwargs  # Pass through any additional parameters
        )
        
        print(f"  ✓ Completed {level_name}")
    
    return df

    
    
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a DataFrame and adjust Nscore based on phenotype data.")
   
    parser.add_argument("--nimbus_table", type=str, help="Path to the input CSV file.")
    parser.add_argument("--results", type=str, help="Path to save the processed CSV file.",default='./results/processed_nimbus_table.csv')
    parser.add_argument('--mode', type=str, default='adjust_Nscore', help="Mode of operation.", choices=['whole_pipline','adjust_Nscore'])
    parser.add_argument("--phenotype_json", type=str,default='./Chains/phenotype_tree.json',help="Path to the phenotype JSON file.")
    args = parser.parse_args()
    
    if args.mode =='whole_pipline': 
        get_response()
        agent = DictionaryAgent()
        with open("./agents/claude_response.txt", "r") as f:
            sys_prompt = f.read()
        result = agent.process_prompt(sys_prompt)
        agent.save_to_json("./Chains/phenotype_tree.json")
        
    df = pd.read_csv(args.nimbus_table)
    processed_df = process_nimbus(df, args.phenotype_json)
    processed_df.to_csv(args.results, index=False)