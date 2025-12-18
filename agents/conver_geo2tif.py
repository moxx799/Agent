import json
import numpy as np
import tifffile
from shapely.geometry import Polygon, Point
from skimage.draw import polygon
import geopandas as gpd
from rasterio.features import rasterize

def geojson_to_instance_tiff(geojson_path, output_path, crop_region=None, separate_nuclei_cells=True):
    """
    Convert QuPath GeoJSON segmentation to instance TIFF file.
    
    Parameters:
    - geojson_path: Path to the GeoJSON file
    - output_path: Path for the output TIFF file
    - crop_region: Tuple (min_x, min_y, max_x, max_y) for cropping, or None for full image
    - separate_nuclei_cells: If True, creates separate instance IDs for nuclei and cells
    """
    
    # Load GeoJSON
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)
    
    features = geojson_data['features'] if 'features' in geojson_data else geojson_data
    
    # First, determine image dimensions by scanning all coordinates
    max_x, max_y = 0, 0
    for feature in features:
 
        cell_coords = feature['geometry']['coordinates'][0]
        for x, y in cell_coords:
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        # Nucleus coordinates, if present
   

    
    print(f"Detected image dimensions: {max_x}x{max_y}")
    
    # Set crop region if not provided
    if crop_region is None:
        crop_region = (0, 0, max_x, max_y)
    
    crop_min_x, crop_min_y, crop_max_x, crop_max_y = crop_region
    crop_width = crop_max_x - crop_min_x
    crop_height = crop_max_y - crop_min_y
    
    print(f"Crop region: {crop_region}")
    print(f"Crop dimensions: {crop_width}x{crop_height}")
    
    # Create instance masks
    cell_instance_mask = np.zeros((crop_height, crop_width), dtype=np.int32)
    nuclei_instance_mask = np.zeros((crop_height, crop_width), dtype=np.int32)
    
    # Process each cell feature
    cell_instance_id = 1
    nuclei_instance_id = 1
    
    for feature in features:
        if feature['properties'].get('objectType') != 'cell':
            continue

        cell_coords = feature['geometry']['coordinates'][0]
        
        # Convert to numpy array and adjust for cropping
        cell_coords = np.array(cell_coords)
        cell_coords[:, 0] -= crop_min_x
        cell_coords[:, 1] -= crop_min_y
        
        # Create mask for cell polygon
        rr, cc = polygon(cell_coords[:, 1], cell_coords[:, 0], 
                        cell_instance_mask.shape)
        
        # Ensure coordinates are within bounds
        valid_mask = (rr >= 0) & (rr < crop_height) & (cc >= 0) & (cc < crop_width)
        rr_valid = rr[valid_mask]
        cc_valid = cc[valid_mask]
        
        if len(rr_valid) > 0:
            cell_instance_mask[rr_valid, cc_valid] = cell_instance_id
        

        if 'nucleusGeometry' in feature:
            nucleus_coords = feature['nucleusGeometry']['coordinates'][0]
            
            # Convert to numpy array and adjust for cropping
            nucleus_coords = np.array(nucleus_coords)
            nucleus_coords[:, 0] -= crop_min_x
            nucleus_coords[:, 1] -= crop_min_y
            
            # Create mask for nucleus polygon
            rr, cc = polygon(nucleus_coords[:, 1], nucleus_coords[:, 0],
                        nuclei_instance_mask.shape)
            
            # Ensure coordinates are within bounds
            valid_mask = (rr >= 0) & (rr < crop_height) & (cc >= 0) & (cc < crop_width)
            rr_valid = rr[valid_mask]
            cc_valid = cc[valid_mask]
            
            if len(rr_valid) > 0:
                nuclei_instance_mask[rr_valid, cc_valid] = nuclei_instance_id

            nuclei_instance_id += 1

        cell_instance_id += 1
    
    print(f"Processed {cell_instance_id-1} cell instances")
    print(f"Cell instance mask unique IDs: {np.unique(cell_instance_mask)}")
    print(f"Nuclei instance mask unique IDs: {np.unique(nuclei_instance_mask)}")
    
    # Save results based on separation requirement
    if separate_nuclei_cells:
        # Save separate TIFF files
        base_path = output_path.replace('.tif', '').replace('.tiff', '')
        
        cell_output_path = f"{base_path}_cells.tif"
        tifffile.imwrite(cell_output_path, cell_instance_mask.astype(np.int32))
        print(f"Cell instances saved to: {cell_output_path}")
        
        nuclei_output_path = f"{base_path}_nuclei.tif"
        tifffile.imwrite(nuclei_output_path, nuclei_instance_mask.astype(np.int32))
        print(f"Nuclei instances saved to: {nuclei_output_path}")
    else:
        # Combine into single TIFF (cells have positive IDs, nuclei have negative IDs)
        combined_mask = cell_instance_mask.copy()
        nuclei_mask_nonzero = nuclei_instance_mask > 0
        combined_mask[nuclei_mask_nonzero] = -nuclei_instance_mask[nuclei_mask_nonzero]
        
        tifffile.imwrite(output_path, combined_mask.astype(np.int32))
        print(f"Combined instances saved to: {output_path}")
    
    return cell_instance_mask, nuclei_instance_mask

def get_crop_region_from_geojson(geojson_path, margin=0):
    """
    Helper function to get crop region from GeoJSON annotation.
    
    Parameters:
    - geojson_path: Path to GeoJSON file
    - margin: Additional margin around the crop region
    """
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)
    
    features = geojson_data['features'] if 'features' in geojson_data else geojson_data
    
    # Find the first annotation feature
    for feature in features:
        if feature['properties'].get('objectType') == 'annotation':
            coords = feature['geometry']['coordinates'][0]
            xs = [coord[0] for coord in coords]
            ys = [coord[1] for coord in coords]
            
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            return (int(min_x - margin), int(min_y - margin), 
                    int(max_x + margin), int(max_y + margin))
    
    return None

# Example usage
if __name__ == "__main__":
    # Method 1: Auto-detect crop region from annotation
    geojson_file = "segmentations.geojson"
    
    # # Get crop region automatically
    # crop_region = get_crop_region_from_geojson(geojson_file, margin=10)
    
    # # Convert to instance TIFF
    # cell_mask, nuclei_mask = geojson_to_instance_tiff(
    #     geojson_file,
    #     "output_instances.tif",
    #     crop_region=crop_region,
    #     separate_nuclei_cells=True
    # )
    
    # Method 2: Manual crop region
    manual_crop = (17000, 4700, 19048, 6748)  # (min_x, min_y, max_x, max_y)
    cell_mask, nuclei_mask = geojson_to_instance_tiff(
        geojson_file,
        "output_instances_manual_crop_normbrain2048.tif",
        crop_region=manual_crop,
        separate_nuclei_cells=True
    )