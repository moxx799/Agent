#!/usr/bin/env python3
"""
tiff_instance_contours.py

Convert a labeled instance-segmentation TIFF (integer labels per instance)
into a binary contour image. Optionally export per-instance contours as a
multi-page TIFF. Works for 2D images or 3D stacks (processes each slice).

Usage:
  python tiff_instance_contours.py input.tif output_contours.tif \
      --thickness 1 --separate-out separate_instances.tif

Dependencies:
  pip install numpy tifffile scikit-image
"""

from pathlib import Path
import argparse
import numpy as np
from tifffile import imread, imwrite
from skimage.segmentation import find_boundaries
from skimage.morphology import dilation, square, ball, disk
from skimage.morphology import binary_erosion

def contour_from_labels_2d(label2d: np.ndarray, thickness: int = 1) -> np.ndarray:
    """
    Create a binary contour image from a 2D label image.

    Strategy:
      - Use skimage.segmentation.find_boundaries on the label map to get
        boundaries between instances and background (and between instances).
      - Optionally dilate to control thickness.

    Returns:
      np.uint8 array with values {0, 255}
    """
    # Boundaries between labels: True at borders of instances (inner borders).
    # mode='inner' avoids marking outside background border pixels redundantly.
    bounds = find_boundaries(label2d, mode='inner')

    if thickness > 1:
        # Make a structuring element (odd size) roughly circular in 2D
        selem = disk(max(1, thickness // 2))
        bounds = dilation(bounds, selem)

    return (bounds.astype(np.uint8) * 255)

def separate_instance_contours_2d(label2d: np.ndarray, thickness: int = 1) -> list:
    """
    Produce one binary contour mask per instance label in a 2D label image.

    More exact per-instance contour is computed by eroding each instance mask
    and XOR with the original (classic morphological boundary extraction),
    then optional dilation to control thickness.
    """
    out = []
    labels = np.unique(label2d)
    labels = labels[labels != 0]  # assume 0 = background

    if thickness > 1:
        selem_dilate = disk(max(1, thickness // 2))
    else:
        selem_dilate = None

    for lab in labels:
        mask = (label2d == lab)
        if not mask.any():
            continue
        # Boundary by xor with its erosion (handles single-object outline precisely)
        # Protect against eroding away tiny objects by using binary_erosion on boolean.
        eroded = binary_erosion(mask)
        contour = np.logical_and(mask, np.logical_not(eroded))
        if selem_dilate is not None:
            contour = dilation(contour, selem_dilate)
        out.append((contour.astype(np.uint8) * 255))
    return out

def process_2d(label2d: np.ndarray, thickness: int, separate_out: bool):
    all_contours = contour_from_labels_2d(label2d, thickness)
    separate = None
    if separate_out:
        separate = separate_instance_contours_2d(label2d, thickness)
    return all_contours, separate

def process_3d(label3d: np.ndarray, thickness: int, separate_out: bool):
    """
    Process a (Z, Y, X) stack slice-by-slice in 2D manner.
    (This avoids accidental 3D connectivity surprises; it’s what most users expect.)
    """
    z, y, x = label3d.shape
    stack_contours = np.zeros((z, y, x), dtype=np.uint8)
    separate_stacks = []  # list of lists; each z has a list of contours

    for i in range(z):
        contours2d, separate2d = process_2d(label3d[i], thickness, separate_out)
        stack_contours[i] = contours2d
        if separate_out:
            separate_stacks.append(separate2d)

    return stack_contours, separate_stacks if separate_out else None

def main(args):


    labels = imread(str(args.input))
    if labels.dtype.kind not in "iu":
        # coerce to integer if some float image slipped in
        labels = labels.astype(np.int64)

    thickness = max(1, int(args.thickness))

    if labels.ndim == 2:
        all_contours, separate = process_2d(labels, thickness, args.separate_out is not None)
        imwrite(str(args.output), all_contours, compression="zlib")

        if args.separate_out is not None:
            if separate:
                # stack pages in order of label value
                stack = np.stack(separate, axis=0)
                imwrite(str(args.separate_out), stack, compression="zlib")
            else:
                # no instances found
                imwrite(str(args.separate_out), np.zeros((1,)+labels.shape, np.uint8), compression="zlib")

    elif labels.ndim == 3:
        all_contours, separate = process_3d(labels, thickness, args.separate_out is not None)
        imwrite(str(args.output), all_contours, compression="zlib")

        if args.separate_out is not None:
            # Flatten per-slice lists into a single multipage stack:
            pages = []
            for slc_list in separate or []:
                pages.extend(slc_list or [])
            if pages:
                stack = np.stack(pages, axis=0)
            else:
                stack = np.zeros((1,) + labels.shape[1:], np.uint8)
            imwrite(str(args.separate_out), stack, compression="zlib")
    else:
        raise ValueError(f"Expected 2D or 3D TIFF. Got shape {labels.shape}")

    print(f"Saved contours to: {args.output}")
    if args.separate_out is not None:
        print(f"Saved per-instance contours to: {args.separate_out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert labeled instances TIFF to binary contours.")
    p.add_argument("--input", type=Path, help="Input labeled TIFF (2D or 3D).")
    p.add_argument("--output", type=Path, help="Output binary contours TIFF (same dimensionality).")
    p.add_argument("--thickness", type=int, default=1,
                   help="Contour thickness in pixels (approx). Default: 1")
    p.add_argument("--separate-out", type=Path, default=None,
                   help="Optional: path to write per-instance contour masks as a multi-page TIFF. "
                        "For 2D input: pages = instances. For 3D input: pages = Z*instances_per_slice.")
    args = p.parse_args()
    main(args)
