import json
import numpy as np
import tifffile
from skimage.measure import find_contours
from shapely.geometry import Polygon, mapping
import uuid
from rasterio.features import shapes

def instance_tiff_to_geojson(cell_tiff_path, nuclei_tiff_path, output_geojson_path, 
                           crop_region=None, original_coordinates=True):
    """
    Convert instance TIFF files back to QuPath-compatible GeoJSON.
    
    Parameters:
    - cell_tiff_path: Path to cell instance TIFF file
    - nuclei_tiff_path: Path to nuclei instance TIFF file  
    - output_geojson_path: Path for output GeoJSON file
    - crop_region: Tuple (min_x, min_y, max_x, max_y) used for cropping, or None
    - original_coordinates: If True, convert back to original coordinates using crop_region
    """
    
    # Load instance masks
    
    cell_mask = tifffile.imread(cell_tiff_path).squeeze(0)
    nuclei_mask = tifffile.imread(nuclei_tiff_path).squeeze(0)
    
    print(f"Cell mask shape: {cell_mask.shape}, unique IDs: {np.unique(cell_mask)}")
    print(f"Nuclei mask shape: {nuclei_mask.shape}, unique IDs: {np.unique(nuclei_mask)}")
    
    # Get crop region info
    if crop_region is not None:
        crop_min_x, crop_min_y, crop_max_x, crop_max_y = crop_region
        print(f"Using crop region: {crop_region}")
    
    features = []
    
    # Get unique instance IDs (excluding background 0)
    cell_instance_ids = np.unique(cell_mask)
    cell_instance_ids = cell_instance_ids[cell_instance_ids > 0]
    
    print(f"Processing {len(cell_instance_ids)} cell instances...")
    
    for cell_id in cell_instance_ids:
        # Create cell feature
        cell_feature = create_cell_feature(cell_mask, nuclei_mask, cell_id, 
                                         crop_region if original_coordinates else None)
        if cell_feature:
            features.append(cell_feature)
    
    # Create GeoJSON structure
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    # Save GeoJSON
    with open(output_geojson_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    print(f"Successfully converted to GeoJSON: {output_geojson_path}")
    print(f"Created {len(features)} cell features")
    
    return geojson

def create_cell_feature(cell_mask, nuclei_mask, cell_id, crop_region=None):
    """
    Create a QuPath cell feature from instance masks.
    """
    try:
        # Extract cell polygon
        cell_polygon = extract_polygon_from_mask(cell_mask, cell_id)
        if cell_polygon is None:
            return None
            
        # Convert to original coordinates if crop region provided
        if crop_region is not None:
            crop_min_x, crop_min_y, _, _ = crop_region
            cell_polygon = [(x + crop_min_x, y + crop_min_y) for x, y in cell_polygon]
        
        # Extract nucleus polygon
        nucleus_polygon = None
        if cell_id <= np.max(nuclei_mask):  # Check if nucleus exists for this cell
            nucleus_polygon = extract_polygon_from_mask(nuclei_mask, cell_id)
            if nucleus_polygon is not None and crop_region is not None:
                crop_min_x, crop_min_y, _, _ = crop_region
                nucleus_polygon = [(x + crop_min_x, y + crop_min_y) for x, y in nucleus_polygon]
        
        # Create QuPath-compatible feature
        feature = {
            "type": "Feature",
            "id": str(uuid.uuid4()),
            "geometry": {
                "type": "Polygon",
                "coordinates": [cell_polygon]  # Note: coordinates are nested
            },
            "properties": {
                "objectType": "cell",
                "isLocked": False,
                "measurements": [],
                "classification": {
                    "name": "Unclassified",
                    "colorRGB": -1
                }
            }
        }
        
        # Add nucleus geometry if available
        if nucleus_polygon is not None:
            feature["nucleusGeometry"] = {
                "type": "Polygon", 
                "coordinates": [nucleus_polygon]
            }
        
        return feature
        
    except Exception as e:
        print(f"Error creating feature for cell {cell_id}: {e}")
        return None

def extract_polygon_from_mask(mask, instance_id):
    """
    Extract polygon coordinates from instance mask using contour detection.
    """
 
    # Create binary mask for this instance
    binary_mask = (mask == instance_id).astype(np.uint32)
    

    # Find contours - using the most recent skimage approach
    contours = find_contours(binary_mask, level=0.5)
    
    if len(contours) == 0:
        print(binary_mask.shape), print(binary_mask.max())
    
    # Use the largest contour (in case of multiple fragments)
    main_contour = max(contours, key=lambda x: len(x))
    
    # Convert to polygon coordinates (swapping x,y since find_contours returns (row, col))
    polygon = [(float(y), float(x)) for x, y in main_contour]
    
    # Ensure polygon is closed (first and last point should be the same)
    if polygon[0] != polygon[-1]:
        polygon.append(polygon[0])
    
    return polygon
    


def extract_polygon_using_rasterio(mask, instance_id):
    """
    Alternative method using rasterio for polygon extraction.
    """
    try:
        binary_mask = (mask == instance_id).astype(np.uint32
                                                   )
        
        # Use rasterio shapes to extract polygons
        results = (
            {'properties': {'raster_val': 1}, 'geometry': s}
            for i, (s, v) in enumerate(
                shapes(binary_mask, mask=binary_mask > 0, transform=None))
        )
        
        geometries = list(results)
        if not geometries:
            return None
            
        # Get the largest geometry
        largest_geom = max(geometries, key=lambda x: x['geometry'].area)
        polygon_shape = largest_geom['geometry']
        
        # Convert to coordinates format
        if polygon_shape['type'] == 'Polygon':
            coordinates = polygon_shape['coordinates'][0]  # exterior ring
            return [(float(x), float(y)) for x, y in coordinates]
        else:
            # Handle MultiPolygon if needed
            all_coords = []
            for poly in polygon_shape['coordinates']:
                all_coords.extend([(float(x), float(y)) for x, y in poly[0]])
            return all_coords
            
    except Exception as e:
        print(f"Rasterio extraction failed for instance {instance_id}: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Paths to your TIFF files
    cell_tiff_path = "C:/Users/lhuang37/Desktop/AGENT/data/gold_standard_labelled/gold_standard_labelled/mibi_breast/masks/TONIC_TMA10_R1C1_whole_cell.tiff"
    nuclei_tiff_path = "C:/Users/lhuang37/Desktop/AGENT/data/gold_standard_labelled/gold_standard_labelled/mibi_breast/masks/TONIC_TMA10_R1C1_whole_cell.tiff"

    # Use the same crop region you used for conversion
    manual_crop = None  # Must match what you used originally
    
    # Convert back to GeoJSON
    geojson_data = instance_tiff_to_geojson(
        cell_tiff_path=cell_tiff_path,
        nuclei_tiff_path=nuclei_tiff_path,
        output_geojson_path="R1C1.geojson",
        crop_region=manual_crop,
        original_coordinates=True
    )

    # cells_tiff_path = "./data/example_dataset/segmentation/deepcell_output/fov0_whole_cell.tiff"
    # nuclei_tiff_path = "./data/example_dataset/segmentation/deepcell_output/fov0_nuclear.tiff"
