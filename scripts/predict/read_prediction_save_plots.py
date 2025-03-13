import zarr
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import h5py

# Define paths
zarr_path = '/raven/u/alik/code/synful/scripts/predict/output/p_setup51/690000/sample_C.zarr'
raw_path = '/raven/u/alik/code/synful/data/cremi/groundtruth/cremiv01/raw/sample_C.hdf'
output_dir = os.path.join(os.path.dirname(zarr_path), 'plots')
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

def plot_vector_field(ax, vectors, stride=40, color='red'):
    """Plot vector field as arrows on the given axis."""
    h, w = vectors.shape[1:3]  # Get height and width from the vector field
    y, x = np.mgrid[0:h:stride, 0:w:stride]
    
    # Extract y and x components for the 2D image plane
    # Note: The first dimension of vectors is the vector components (z,y,x)
    v = vectors[1, ::stride, ::stride]  # y component
    u = vectors[2, ::stride, ::stride]  # x component
    
    # Plot vectors in their actual size (no normalization)
    ax.quiver(x, y, u, v, color=color, angles='xy', scale_units='xy', scale=1, width=0.003)
    ax.set_aspect('equal')

def save_volume_plots(data, raw_slice, name, output_dir, z_index=None, partner_vectors=None):
    """Save plots for a given volume with raw data overlay."""
    if z_index is None:
        z_index = data.shape[0] // 2
    
    # Get the slice
    if len(data.shape) == 3:
        slice_data = data[z_index]
    else:  # 4D data
        slice_data = data[z_index, 0]  # Assuming first channel
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()  # Flatten axes for easier indexing
    
    # Plot the raw data
    im1 = axes[0].imshow(raw_slice, cmap='gray')
    axes[0].set_title(f'Raw Data (z={z_index})')
    plt.colorbar(im1, ax=axes[0])
    axes[0].set_aspect('equal')
    
    # Plot the prediction
    im2 = axes[1].imshow(slice_data, cmap='viridis')
    axes[1].set_title(f'{name} (z={z_index})')
    plt.colorbar(im2, ax=axes[1])
    axes[1].set_aspect('equal')
    
    # Plot overlay
    # Normalize prediction data to [0,1] for overlay
    pred_normalized = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
    # Create RGB overlay
    overlay = np.zeros((*raw_slice.shape, 3))
    overlay[..., 0] = raw_slice / 255.0  # Red channel - raw data
    overlay[..., 1] = pred_normalized  # Green channel - prediction
    im3 = axes[2].imshow(overlay)
    axes[2].set_title('Overlay (Red: Raw, Green: Prediction)')
    axes[2].set_aspect('equal')
    
    # If we have partner vectors, plot them
    if partner_vectors is not None:
        print(f"Partner vectors shape at this point: {partner_vectors.shape}")  # Debug print
        # Get the vector field for this z-slice
        # partner_vectors has shape (3,Z,Y,X) where first dimension is (z,y,x) components
        vector_slice = partner_vectors[:, z_index]
        # Plot raw data
        axes[3].imshow(raw_slice, cmap='gray')
        # Overlay vector field
        plot_vector_field(axes[3], vector_slice)
        axes[3].set_title(f'Partner Vectors (z={z_index})')
        axes[3].set_aspect('equal')
    
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
    
    # Get partner vectors if available
    partner_vectors = volumes.get('pred_partner_vectors', None)
    if partner_vectors is not None:
        print(f"Partner vectors shape: {partner_vectors.shape}")  # Debug print
    
    for key in volumes.keys():
        if key != 'pred_partner_vectors':  # Skip plotting vectors separately
            data = volumes[key]
            print(f"\nDataset: {key}")
            print(f"Shape: {data.shape}")
            print(f"Dtype: {data.dtype}")
            print(f"Chunks: {data.chunks}")
            
            # Save plots for multiple slices
            for z in [data.shape[0]//4, data.shape[0]//2, 3*data.shape[0]//4]:
                raw_slice = raw_data[z]
                save_volume_plots(data, raw_slice, key, output_dir, z, 
                                partner_vectors[...] if partner_vectors is not None else None)

print(f"\nPlots and statistics have been saved to: {os.path.abspath(output_dir)}") 