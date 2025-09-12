#!/usr/bin/env python3
"""
Demo script for the new DCT+Mamba network structure.
Shows how to use the 'dct_mamba' backend that implements:
1. Spectral branch: DCT + real Mamba processing
2. Spatial branch: real Mamba processing
"""

import torch
import numpy as np
from train import DualBranchUnmixNet, Data, PatchDataset
from torch.utils.data import DataLoader

def create_synthetic_data(H=50, W=50, L=30, K=3):
    """Create synthetic hyperspectral data for testing."""
    print(f"Creating synthetic data: {H}x{W}x{L} with {K} endmembers")
    
    # Create synthetic endmembers (smooth spectral curves)
    endmembers = np.zeros((K, L))
    for k in range(K):
        # Create different spectral signatures
        x = np.linspace(0, 4*np.pi, L)
        endmembers[k] = np.abs(np.sin(x + k*np.pi/3) * np.exp(-x/10)) + 0.1
        endmembers[k] /= np.max(endmembers[k])  # Normalize
    
    # Create synthetic abundance maps (smooth spatial distribution)
    abundances = np.zeros((H, W, K))
    for k in range(K):
        # Create spatial patterns for each endmember
        y, x = np.meshgrid(np.linspace(-2, 2, H), np.linspace(-2, 2, W), indexing='ij')
        if k == 0:
            abundances[:, :, k] = np.exp(-(x**2 + y**2))
        elif k == 1:
            abundances[:, :, k] = np.exp(-((x-1)**2 + (y-1)**2))
        else:
            abundances[:, :, k] = np.exp(-((x+1)**2 + (y+1)**2))
    
    # Normalize abundances to sum to 1
    abundance_sum = abundances.sum(axis=2, keepdims=True)
    abundances = abundances / (abundance_sum + 1e-8)
    
    # Generate hyperspectral data
    hyperspectral = np.zeros((H, W, L))
    for i in range(H):
        for j in range(W):
            hyperspectral[i, j] = abundances[i, j] @ endmembers
    
    # Add small amount of noise
    noise = np.random.normal(0, 0.01, hyperspectral.shape)
    hyperspectral += noise
    hyperspectral = np.clip(hyperspectral, 0, None)
    
    return hyperspectral, abundances, endmembers

def demo_dct_mamba_backend():
    """Demonstrate the DCT+Mamba backend."""
    print("=== DCT+Mamba Backend Demo ===")
    
    # Create synthetic data
    H, W, L, K = 50, 50, 30, 3
    patch_size = 5
    hyperspectral, abundances, endmembers = create_synthetic_data(H, W, L, K)
    
    print(f"Data shapes:")
    print(f"  Hyperspectral: {hyperspectral.shape}")
    print(f"  Abundances: {abundances.shape}")
    print(f"  Endmembers: {endmembers.shape}")
    
    # Convert to tensors
    hs_tensor = torch.from_numpy(hyperspectral.reshape(-1, L)).float()
    abd_tensor = torch.from_numpy(abundances.reshape(-1, K)).float()
    
    # Create a simple data object (mimicking the Data class interface)
    class SimpleData:
        def __init__(self, hs_img, abd_map, endmembers, H, W, L, K):
            self.data = {
                'hs_img': hs_img,
                'abd_map': abd_map,
                'end_mem': torch.from_numpy(endmembers).float(),
                'init_weight': torch.from_numpy(endmembers).float()
            }
            self.H, self.W, self.L, self.K = H, W, L, K
        
        def get(self, key):
            return self.data[key]
        
        def get_L(self):
            return self.L
        
        def get_P(self):
            return self.K
        
        def get_col(self):
            return self.W
    
    data_obj = SimpleData(hs_tensor, abd_tensor, endmembers, H, W, L, K)
    
    # Create dataset and dataloader
    dataset = PatchDataset(data_obj, patch=patch_size, stride=2)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    print(f"Dataset size: {len(dataset)} patches")
    
    # Test different backends
    backends = ['like', 'dct', 'dct_mamba']
    
    for backend in backends:
        print(f"\n--- Testing backend: {backend} ---")
        
        # Create model
        model = DualBranchUnmixNet(
            L=L, K=K, d=64, ls=2, lp=2, patch=patch_size, 
            E0=torch.from_numpy(endmembers).float(),
            backend=backend
        )
        
        print(f"Spectral branch: {type(model.spec).__name__}")
        print(f"Spatial branch: {type(model.spa).__name__}")
        
        # Test forward pass on a batch
        model.eval()
        with torch.no_grad():
            for batch_idx, (patch, y_center, a_true) in enumerate(dataloader):
                if batch_idx > 0:  # Just test first batch
                    break
                
                print(f"Batch shapes: patch={patch.shape}, y_center={y_center.shape}")
                
                # Forward pass
                output = model(patch, y_center)
                
                print(f"Output shapes:")
                for key, val in output.items():
                    print(f"  {key}: {val.shape}")
                
                # Calculate some basic metrics
                reconstruction_error = torch.nn.functional.mse_loss(output['Xhat'], output['Y'])
                abundance_sum = output['A'].sum(dim=1).mean()
                
                print(f"Reconstruction MSE: {reconstruction_error:.6f}")
                print(f"Mean abundance sum: {abundance_sum:.6f}")
                
                break
    
    print("\n=== Demo completed successfully! ===")
    print("\nKey features of the DCT+Mamba backend ('dct_mamba'):")
    print("1. ✓ Spectral branch combines DCT frequency analysis with Mamba modeling")
    print("2. ✓ Spatial branch uses Mamba for spatial relationship modeling")
    print("3. ✓ Maintains compatibility with existing training pipeline")
    print("4. ✓ Suitable for hyperspectral unmixing tasks")

if __name__ == '__main__':
    demo_dct_mamba_backend()