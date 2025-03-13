import os
import numpy as np
import h5py
import tifffile
from pathlib import Path

def convert_bbox_to_hdf5(bbox_dir, output_file, take_every_nth=5):
    """Convert a directory of TIFF files to an HDF5 file with appropriate metadata."""
    # Get all TIFF files and sort them
    tiff_files = sorted([f for f in Path(bbox_dir).glob('*.tif')])
    
    # Take every nth slice
    tiff_files = tiff_files[::take_every_nth]
    
    # Read first image to get dimensions
    first_img = tifffile.imread(tiff_files[0])
    
    # Pre-allocate array for all slices
    volume = np.zeros((len(tiff_files), *first_img.shape), dtype=np.uint8)
    
    # Read all slices
    print(f"Reading {len(tiff_files)} slices from {bbox_dir}...")
    for i, tiff_file in enumerate(tiff_files):
        volume[i] = tifffile.imread(tiff_file)
    
    # Create HDF5 file
    print(f"Creating HDF5 file: {output_file}")
    with h5py.File(output_file, 'w') as f:
        # Create volumes group
        volumes = f.create_group('volumes')
        
        # Create raw dataset
        raw = volumes.create_dataset('raw', data=volume, dtype=np.uint8)
        
        # Set attributes for raw dataset
        raw.attrs.create('resolution', [40, 8, 8], dtype=np.int64)
        raw.attrs.create('offset', [0, 0, 0], dtype=np.int64)

def main():
    # Base directories
    base_input_dir = '/u/alik/code/synful/data/fibsem/raw'
    base_output_dir = '/u/alik/code/synful/data/fibsem/raw'
    
    # Create output directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Process each bbox
    for i in range(1, 8):
        bbox_dir = os.path.join(base_input_dir, f'bbox{i}')
        output_file = os.path.join(base_output_dir, f'bbox{i}.hdf')
        
        print(f"\nProcessing bbox{i}...")
        convert_bbox_to_hdf5(bbox_dir, output_file)
        print(f"Completed bbox{i}")

if __name__ == '__main__':
    main() 