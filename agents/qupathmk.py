import numpy as np
import tifffile
from ome_types import OME, model
from scipy.ndimage import zoom
import os

def generate_pyramid(base_image, num_levels, axes, dtype=None):
    """Generate pyramid levels with exact level count matching reference"""
    pyramid = [base_image]
    current = base_image
    if dtype is None:
        for _ in range(num_levels - 1):
            zoom_factors = [0.5 if ax in 'YX' else 1 for ax in axes]
            downsampled = zoom(current, zoom_factors, order=1)
            pyramid.append(downsampled)
            del current
            current = downsampled
    else:
        for _ in range(num_levels - 1):
            zoom_factors = [0.5 if ax in 'YX' else 1 for ax in axes]
            downsampled = zoom(current, zoom_factors, order=1).astype(dtype)
            pyramid.append(downsampled)
            del current
            current = downsampled
    
    return pyramid # Strictly enforce level count


#**********************************************************************************
def construct(output_file, asf_path, ref_levels, ref_axes, dimention_order, datatype,reference_pixel_size,marker_dict):
    biomarkers = list(marker_dict.keys())
    print(biomarkers)
    # Load and stack channels,
    processed_channels = []
    for biomarker in biomarkers:
        print('Checking:', biomarker)
        # channel_path = os.path.join(asf_path, f"S1_{channel_dict[biomarker]}.tif")
        # channel_path = os.path.join(asf_path, f"{cell_type_panel[biomarker]}")
        file_name = marker_dict[biomarker]
        channel_path = os.path.join(asf_path, file_name)
        single_channel = tifffile.imread(channel_path)
        if single_channel.max() <= 1:
            if datatype == np.uint16:
                single_channel = single_channel * 65535
            elif datatype == np.uint8:
                single_channel = single_channel * 255
            else:
                raise ValueError("Unsupported datatype for scaling.")
        processed_channels.append(single_channel.astype(datatype))
        

    c_axis = ref_axes.index('C')
    processed_base = np.stack(processed_channels, axis=c_axis)

    # Generate pyramid with EXACTLY ref_levels levels
    processed_pyramid = generate_pyramid(processed_base, ref_levels, ref_axes, datatype)
    print('pyramid shape:', [level.shape for level in processed_pyramid])

    # Create OME metadata
    new_ome = OME()
    ome_channels = [model.Channel(id=f"Channel:0:{i}", name=name,samples_per_pixel=1) 
                for i, name in enumerate(biomarkers)]

    x_dim = ref_axes.index('X')
    y_dim = ref_axes.index('Y')
    size_x = processed_base.shape[x_dim]
    size_y = processed_base.shape[y_dim]

    new_pixels = model.Pixels(
        dimension_order=dimention_order,
        type=processed_base.dtype.name,
        size_c=len(biomarkers),
        size_z=1,
        size_t=1,
        size_x=size_x,
        size_y=size_y,
        channels=ome_channels,
        physical_size_x=REFERENCE_PIXEL_SIZE,
        physical_size_y=REFERENCE_PIXEL_SIZE,
    )
    del processed_base
    new_ome.images.append(model.Image(
        id="Image:0",
        name="Generated Image",
        pixels=new_pixels
    ))

    # Write with strict pyramid validation
    with tifffile.TiffWriter(output_file, bigtiff=True) as tif:
        # Write base level
        tif.write(
            processed_pyramid[0],
            photometric='minisblack',
            description=new_ome.to_xml(),
            compression='zlib',
            subifds=len(processed_pyramid)-1,

        )
        
        # Write pyramid levels
        print('total levels:', len(processed_pyramid))
        for i, level in enumerate(processed_pyramid[1:]):
            if level.size == 0:  # Skip empty placeholder levels
                continue
            print(f"Processing level {i+1} ")
            tif.write(
                level,
                photometric='minisblack',
                subfiletype=1,
                compression='zlib',
            )
            processed_pyramid[i+1] = None
            
import argparse

argparse = argparse.ArgumentParser()
argparse.add_argument('--output', type=str, help='Output OME-TIFF file path')
argparse.add_argument('--asf_path', type=str, help='Path to the directory containing channel images')
argparse.add_argument('--ref_levels', type=int, default= 8 ,help='Number of pyramid levels in the reference image')
argparse.add_argument('--ref_axes', type=str, default= 'CYX',help='Axes order of the reference image (e.g., CYX)')
argparse.add_argument('--dimention_order', type=str, default= 'XYCZT', help='Dimension order for OME metadata (e.g., XYCZT)')
argparse.add_argument('--datatype', type=str, default= 'uint16', help='Data type for the image (e.g., uint16)')
argparse.add_argument('--pixel_size', type=float, default= 330 , help='Pixel size in nm (e.g., 330 for 0.33 µm)')
argparse.add_argument('--mode', type=str, default= 'combine' , help="Mode: 'combine' or 'separate'")

args = argparse.parse_args()
            
# User inputs
PIXEL_SIZE_NM = args.pixel_size  # Pixel size in nanometers
REFERENCE_PIXEL_SIZE = PIXEL_SIZE_NM / 1000  # Convert nm to µm
datatype = args.datatype

datatype = np.dtype(datatype)


print('Using datatype:', datatype)
# marker_dict =  {
#     "R1C2.tif": "DAPI",
#     "R1C10.tif": "NFH",
#     "R1C3.tif": "CC3",
#     "R1C4.tif": "NeuN",
#     "R1C5.tif": "MBP",
#     "R1C6.tif": "RECA1",
#     "R1C7.tif": "IBA1++",  
#     "R1C8.tif": "TomatoLectin",
#     "R1C9.tif": "PCNA",
#     "R2C10.tif": "MAP2",
#     "R2C3.tif": "GAD67",
#     "R2C4.tif": "GFAP",
#     "R2C5.tif": "Parvalbumin",
#     "R2C6.tif": "S100",
#     "R2C7.tif": "Calretinin",
#     "R2C8.tif": "TomatoLectin",
#     "R2C9.tif": "CD31"
# }

# marker_dict =  {
#     "R1C4.tif": "NeuN",
#     "R1C6.tif": "RECA1",
#     "R1C7.tif": "IBA1++",  
#     "R2C4.tif": "GFAP",
#     "R2C6.tif": "S100",
# }
# CD14 ,CD163 ,CD20 ,CD3
# ,CD31 ,CD4 ,CD45 ,CD68 ,CD8 ,CK17 ,
# Collagen1 ,ECAD ,Fibronectin ,GLUT1 ,HLADR ,IDO ,Ki67 ,PD1 ,SMA ,Vim ,
# marker_dict =  {
#     "CD3.tiff": "CD3",
#     "CD4.tiff": "CD4",
#     "CD8.tiff": "CD8",
#     "CD20.tiff": "CD20",
#     "CD45.tiff": "CD45",
#     "CD68.tiff": "CD68",
#     "CK17.tiff": "CK17",
#     "Collagen1.tiff": "Collagen1",
#     "ECAD.tiff": "ECAD",
#     "Fibronectin.tiff": "Fibronectin",
#     "GLUT1.tiff": "GLUT1",
#     "HLADR.tiff": "HLADR",
#     "IDO.tiff": "IDO",
#     "Ki67.tiff": "Ki67",
#     "PD1.tiff": "PD1",
#     "SMA.tiff": "SMA",
#     "Vim.tiff": "Vim",
#     'CD14.tiff': "CD14",
#     'CD163.tiff': "CD163",
#     'CD31.tiff': "CD31",
# }
files = os.listdir(args.asf_path)
marker_dict = {}
for f in files:
    if f.endswith('.tiff'):
        marker_dict[f] = f.split('.')[0]

# revert the marker_dict
marker_dict = {v: k for k, v in marker_dict.items()}

mode = args.mode
if mode == 'combine':

    construct(args.output, args.asf_path, args.ref_levels, args.ref_axes, args.dimention_order, datatype,REFERENCE_PIXEL_SIZE,marker_dict)

elif mode == 'separate':
    for k,v in marker_dict.items():
        biomarker = {k,v}
        output_file = os.path.join(os.path.dirname(args.output), f"{v}.ome.tiff")
        construct(output_file, args.asf_path, args.ref_levels, args.ref_axes, args.dimention_order, datatype,REFERENCE_PIXEL_SIZE,marker_dict)

else:
    raise ValueError("Invalid mode selected. Choose 'combine' or 'separate'.")