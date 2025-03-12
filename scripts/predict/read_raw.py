import os
import h5py
from PIL import Image
import numpy as np

# Define paths
hdf5_file_path = 'data/cremi/groundtruth/cremiv01/raw/sample_C.hdf'
tiff_output_dir = 'data/cremi/groundtruth/cremiv01/raw/sample_C/tiff/'

# First, let's inspect the HDF5 file structure
with h5py.File(hdf5_file_path, 'r') as hdf5_file:
    # Print all groups and datasets in the file
    print("HDF5 file structure:")
    print("Keys:", list(hdf5_file.keys()))
    
    # Print more detailed information about each key
    def print_structure(name, obj):
        print(f"Name: {name}")
        if isinstance(obj, h5py.Dataset):
            print(f"  Type: Dataset")
            print(f"  Shape: {obj.shape}")
            print(f"  Dtype: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"  Type: Group")
    
    hdf5_file.visititems(print_structure)

print("\nFile inspection complete.")

# Create the output directory if it doesn't exist
os.makedirs(tiff_output_dir, exist_ok=True)

# Open the HDF5 file
with h5py.File(hdf5_file_path, 'r') as hdf5_file:
    # Get the dataset from volumes/raw
    dataset = hdf5_file['volumes/raw']
    print(f"Dataset shape: {dataset.shape}")
    print(f"Dataset dtype: {dataset.dtype}")
    
    # Iterate over each slice in the dataset
    for i in range(dataset.shape[0]):
        # Read the slice
        slice_data = dataset[i, :, :]
        
        # Convert to uint8 if not already
        if slice_data.dtype != np.uint8:
            slice_data = slice_data.astype(np.uint8)
        
        # Convert to an image
        image = Image.fromarray(slice_data)
        
        # Save as TIFF
        output_path = os.path.join(tiff_output_dir, f'slice_{i:04d}.tiff')
        image.save(output_path)
        
        # Print progress every 10 slices
        if i % 10 == 0:
            print(f"Processed slice {i}/{dataset.shape[0]}")

print("Conversion complete.")