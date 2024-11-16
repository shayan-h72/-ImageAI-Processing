3D to 2D Image Slicing Script

This Python script takes a directory of 3D NIfTI images (.nii or .nii.gz format) and splits each 3D image into multiple 2D slices. 
The slices are saved in TIFF format (.tif). The user can specify the axis along which to slice the image (0, 1, or 2), 
and the script allows selecting a range of indices from the 3D volume to avoid empty regions. This is useful for processing 
and visualizing medical image datasets, such as MRI scans, where individual slices are often analyzed.

Features:

- Input: Directory containing 3D NIfTI images.
- Output: Directory to save the resulting 2D slices as TIFF images.
- Customizable options:
    - Choose the axis (0, 1, or 2) along which the image is sliced.
    - Specify a percentage range to sample slices, avoiding empty sections of the volume.

Requirements:

- Python 3.x
- argparse, glob, os, PIL, nibabel, numpy

Usage:

python split_3d_to_2d.py <input_directory> <output_directory> [-a <axis>] [-p <start_pct> <end_pct>]


- `input_directory`: Path to the directory containing NIfTI files.
- `output_directory`: Path to the directory where the 2D slices will be saved.
- `-a` (optional): Axis for slicing (0, 1, or 2). Default is 2 (slicing along the third dimension).
- `-p` (optional): Range of indices to slice, as a percentage. Default is from 20% to 80% of the axis range.

Example:

python split_3d_to_2d.py ./input_images ./output_slices -a 2 -p 0.2 0.8


This command processes all .nii files in the ./input_images directory, slices them along the third axis (axis 2), 
and saves the slices from 20% to 80% of the axis range in ./output_slices.

