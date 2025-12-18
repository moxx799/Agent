import tifffile
import pandas as pd
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt

channel_colors = {  'R2C6': [0, 255, 0],      # Green
                     'R1C6': [255, 0, 0],      # Red
                     'unknown': [128, 128, 128],  # Gray
                     'ambiguous': [0, 0, 255]    # Blue
                  }

    # Create legend
# channel_colors = {
#     'R1C4': [255, 0, 0],      # Red
#     'R1C6': [0, 255, 0],      # Green  
#     'R1C7': [0, 0, 255],      # Blue
#     'R2C4': [255, 255, 0],    # Yellow
#     'R2C6': [255, 0, 255]     # Magenta
# }


def color_instance_mask_by_channel(mask_path,table, output_path, channel_colors):
    """
    Color instance masks based on positive_channel values from a table
    
    Args:
        mask_path: Path to the instance mask TIFF file
        table_path: Path to the table file (CSV or Excel)
        output_path: Path to save the colored output TIFF
    """
    
    # Read the instance mask
    instance_mask = tifffile.imread(mask_path)
    
    # Read the table (adjust based on your file format)
    # For CSV: pd.read_csv(table_path)
    # For Excel: pd.read_excel(table_path)
    # Create a dictionary mapping labels to positive_channel
    label_to_channel = dict(zip(table['label'], table['positive_channel']))
    
    # Define colors for each channel

    
    # Create an empty RGB image
    colored_mask = np.zeros((*instance_mask.shape, 3), dtype=np.uint8)
    
    # Get unique labels from the mask
    unique_labels = np.unique(instance_mask)
    
    for label in unique_labels:
        if label == 0:  # Skip background
            continue
            
        if label in label_to_channel:
            channel = label_to_channel[label]
            if channel in channel_colors:
                # Create mask for this instance
                instance_region = (instance_mask == label)
                
                # Apply color to this instance
                for channel_idx in range(3):
                    colored_mask[instance_region, channel_idx] = channel_colors[channel][channel_idx]
            else:
                print(f"Warning: Channel {channel} not found in color mapping for label {label}")
        else:
            print(f"Warning: Label {label} not found in table")
    
    # Save the colored mask
    tifffile.imwrite(output_path, colored_mask)
    print(f"Colored mask saved to: {output_path}")
    
    return colored_mask

def visualize_colored_mask(colored_mask, table, channel_colors):
    """
    Visualize the colored mask with a legend
    """
    plt.figure(figsize=(12, 10))
    
    # Display the colored mask
    plt.imshow(colored_mask)
    plt.title('Colored Instance Mask by Positive Channel')
    plt.axis('off')
    

    # Create legend patches
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=np.array(color)/255, 
                           label=channel) 
                     for channel, color in channel_colors.items()]
    
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.show()

# Alternative version with more detailed statistics
def color_instance_mask_with_stats(mask_path,table, output_path, channel_colors):
    """
    Color instance masks with detailed statistics
    """
    # Read data
    instance_mask = tifffile.imread(mask_path)
   # Adjust based on your file format
    
    # Create mapping and colors
    label_to_channel = dict(zip(table['label'], table['positive_channel']))
    

    # Initialize colored mask and statistics
    colored_mask = np.zeros((*instance_mask.shape, 3), dtype=np.uint8)
    stats = {channel: 0 for channel in channel_colors.keys()}
    unmapped_labels = []
    
    # Process each label
    unique_labels = np.unique(instance_mask)
    
    for label in unique_labels:
        if label == 0:  # Skip background
            continue
            
        if label in label_to_channel:
            channel = label_to_channel[label]
            if channel in channel_colors:
                instance_region = (instance_mask == label)
                color = channel_colors[channel]
                
                # Apply color
                for channel_idx in range(3):
                    colored_mask[instance_region, channel_idx] = color[channel_idx]
                
                stats[channel] += 1
            else:
                unmapped_labels.append(label)
        else:
            unmapped_labels.append(label)
    
    # Print statistics
    print("Instance coloring statistics:")
    print(f"Total instances processed: {len(unique_labels) - 1}")  # Exclude background
    print("Instances per channel:")
    for channel, count in stats.items():
        print(f"  {channel}: {count} instances")
    
    if unmapped_labels:
        print(f"Unmapped labels: {unmapped_labels}")
    
    # Save the result
    tifffile.imwrite(output_path, colored_mask)
    print(f"Colored mask saved to: {output_path}")
    
    return colored_mask, stats


    # Replace these paths with your actual file paths
# mask_file = "C:/Users/lhuang37/Desktop/AGENT/data/gold_standard_labelled/gold_standard_labelled/mibi_breast/masks/TONIC_TMA10_R3C6_whole_cell.tiff"
mask_file = "C:/Users/lhuang37/Desktop/AGENT/data/tbi/segmentation/deepcell_output/fov0_whole_cell.tiff"

output_file = "./R1C6_colored_maskp2.tiff"
colored_mask = color_instance_mask_by_channel(mask_file, df, output_file, channel_colors)

# Method 2: With statistics
# colored_mask, stats = color_instance_mask_with_stats(mask_file, table_file, output_file)

# Visualize the result

# visualize_colored_mask(colored_mask, df, channel_colors)
