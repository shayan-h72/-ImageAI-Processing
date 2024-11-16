import argparse
from glob import glob
import os
import sys
from PIL import Image
import nibabel as nib
import numpy as np

def arg_parser():
    parser = argparse.ArgumentParser(description='split 3d image into multiple 2d images')
    parser.add_argument('img_dir', type=str, help='path to nifti image directory')
    parser.add_argument('out_dir', type=str, help='path to output the corresponding tif image slices')
    parser.add_argument('-a', '--axis', type=int, default=2, help='axis of the 3d image array on which to sample the slices')
    parser.add_argument('-p', '--pct-range', nargs=2, type=float, default=(0.2,0.8),help=('range of indices, as a percentage, from which to sample in each 3d image volume. used to avoid creating blank tif images if there is substantial empty space along the ends of the chosen axis'))
    return parser
