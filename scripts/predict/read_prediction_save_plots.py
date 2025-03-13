import zarr
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import h5py

# Define paths
zarr_path = '/raven/u/alik/code/synful/scripts/predict/output/p_setup52/390000/sample_C.zarr'
raw_path = '/raven/u/alik/code/synful/data/cremi/groundtruth/cremiv01/raw/sample_C.hdf'
output_dir = os.path.dirname(zarr_path)
os.makedirs(output_dir, exist_ok=True)

# Open the zarr file
f = zarr.open(zarr_path, 'r')

# Open the raw data
with h5py.File(raw_path, 'r') as raw_file:
    raw_data = raw_file['volumes/raw'][:]

# Print basic information about the file structure
print("=== Zarr File Structure ===")
print("Arrays:", list(f.keys()))
print("Groups:", [k for k in f.group_keys()])

def save_volume_plots(data, raw_slice, name, output_dir, z_index=None):
    """Save plots for a given volume with raw data overlay."""
    if z_index is None:
        z_index = data.shape[0] // 2
    
    # Get the slice
    if len(data.shape) == 3:
        slice_data = data[z_index]
    else:  # 4D data
        slice_data = data[z_index, 0]  # Assuming first channel
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    
    # Plot the raw data
    im1 = ax1.imshow(raw_slice, cmap='gray')
    ax1.set_title(f'Raw Data (z={z_index})')
    plt.colorbar(im1, ax=ax1)
    
    # Plot the prediction
    im2 = ax2.imshow(slice_data, cmap='viridis')
    ax2.set_title(f'{name} (z={z_index})')
    plt.colorbar(im2, ax=ax2)
    
    # Plot overlay
    # Normalize prediction data to [0,1] for overlay
    pred_normalized = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
    # Create RGB overlay
    overlay = np.zeros((*raw_slice.shape, 3))
    overlay[..., 0] = raw_slice / 255.0  # Red channel - raw data
    overlay[..., 1] = pred_normalized  # Green channel - prediction
    im3 = ax3.imshow(overlay)
    ax3.set_title('Overlay (Red: Raw, Green: Prediction)')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, f'{name}_z{z_index}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save statistics to text file
    stats = {
        'min': np.min(slice_data),
        'max': np.max(slice_data),
        'mean': np.mean(slice_data),
        'std': np.std(slice_data),
        'non_zero': np.count_nonzero(slice_data)
    }
    
    with open(os.path.join(output_dir, f'{name}_z{z_index}_stats.txt'), 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value:.4f}\n")
        
    print(f"\nStatistics for {name} (z={z_index}):")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")

# Process each volume
if 'volumes' in f:
    volumes = f['volumes']
    print("\n=== Volumes Contents ===")
    print("Available datasets:", list(volumes.keys()))
    
    for key in volumes.keys():
        data = volumes[key]
        print(f"\nDataset: {key}")
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
        print(f"Chunks: {data.chunks}")
        
        # Save plots for multiple slices
        for z in [data.shape[0]//4, data.shape[0]//2, 3*data.shape[0]//4]:
            raw_slice = raw_data[z]
            save_volume_plots(data, raw_slice, key, output_dir, z)

print(f"\nPlots and statistics have been saved to: {os.path.abspath(output_dir)}") 