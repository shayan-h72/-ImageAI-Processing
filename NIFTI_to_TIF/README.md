# 3D to 2D Image Slicing Script

This Python script processes a directory of 3D NIfTI images (`.nii` or `.nii.gz` format) and splits each 3D image into multiple 2D slices, saving them in TIFF format (`.tif`). You can customize the axis along which the image is sliced (0, 1, or 2), and select a range of indices from the 3D volume to avoid processing empty regions. This is particularly useful for visualizing and analyzing medical image datasets, such as MRI scans, where individual slices are analyzed.

---

## Features

- **Input**: Directory containing 3D NIfTI images.
- **Output**: Directory to save the resulting 2D slices as TIFF images.
- **Customizable options**:
  - Choose the axis (0, 1, or 2) along which the image is sliced.
  - Specify a percentage range to sample slices, avoiding empty sections of the volume.

---

## Requirements

- Python 3.x
- Required Python packages:
  - `argparse`
  - `glob`
  - `os`
  - `PIL` (Pillow)
  - `nibabel`
  - `numpy`

---

## Usage

To run the script, use the following command:

```bash
python convrt_nifti_to_tif.py <input_directory> <output_directory> [-a <axis>] [-p <start_pct> <end_pct>]
```

This will split each 3D image along the specified axis and save the selected slices in the desired output directory.

## Arguments:
- `input_directory`: Path to the directory containing NIfTI files.
- `output_directory`: Path to the directory where the 2D slices will be saved.
- `a` (optional): Axis for slicing (0, 1, or 2). Default is 2 (slicing along the third dimension).
- `p` (optional): Range of indices to slice, as a percentage. Default is from 20% to 80% of the axis range.

## Example
To process all `.nii` files in the `./input_images` directory, slice them along the third axis (`axis 2`), and save the slices from 20% to 80% of the axis range in the `./output_slices` directory, use the following command:

```bash
python split_3d_to_2d.py ./input_images ./output_slices -a 2 -p 0.2 0.8
```
