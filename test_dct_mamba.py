#!/usr/bin/env python3
"""
Test script for the new DCT+Mamba network structure.
Verifies that the implementation meets the requirements:
1. Spectral branch uses DCT + real Mamba processing
2. Spatial branch uses real Mamba processing
"""

import torch
import numpy as np
from train import DualBranchUnmixNet, SpectralDCTMamba, SpatialMambaReal, SpatialMambaLike, HAS_MAMBA

def test_dct_mamba_backend():
    """Test the new DCT+Mamba backend implementation."""
    print("=== Testing DCT+Mamba Backend ===")
    
    # Test parameters
    L, K, patch = 20, 4, 5  # L=20 bands, K=4 endmembers, 5x5 patch
    B = 4  # Batch size
    d = 64  # Feature dimension
    
    print(f"Test setup: L={L}, K={K}, patch={patch}, B={B}, d={d}")
    print(f"mamba-ssm available: {HAS_MAMBA}")
    
    # Create model with DCT+Mamba backend
    model = DualBranchUnmixNet(L=L, K=K, d=d, backend='dct_mamba', patch=patch)
    
    # Verify the branch types
    print(f"\nSpectral branch type: {type(model.spec).__name__}")
    print(f"Spatial branch type: {type(model.spa).__name__}")
    
    # Check that spectral branch is DCT+Mamba
    assert isinstance(model.spec, SpectralDCTMamba), f"Expected SpectralDCTMamba, got {type(model.spec)}"
    
    # Check that spatial branch is the appropriate Mamba implementation
    if HAS_MAMBA:
        assert isinstance(model.spa, SpatialMambaReal), f"Expected SpatialMambaReal when mamba-ssm available, got {type(model.spa)}"
    else:
        assert isinstance(model.spa, SpatialMambaLike), f"Expected SpatialMambaLike when mamba-ssm unavailable, got {type(model.spa)}"
    
    print("✓ Branch types verified correctly")
    
    # Test forward pass
    test_patch = torch.randn(B, L, patch, patch)
    test_y = torch.randn(B, L)
    
    print(f"\nTesting forward pass...")
    print(f"Input patch shape: {test_patch.shape}")
    print(f"Input center pixel shape: {test_y.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(test_patch, test_y)
    
    # Verify outputs
    expected_keys = ['A', 'E', 'Xhat', 'Y']
    assert set(output.keys()) == set(expected_keys), f"Expected keys {expected_keys}, got {list(output.keys())}"
    
    assert output['A'].shape == (B, K), f"Expected abundance shape {(B, K)}, got {output['A'].shape}"
    assert output['E'].shape == (K, L), f"Expected endmember shape {(K, L)}, got {output['E'].shape}"
    assert output['Xhat'].shape == (B, L), f"Expected reconstruction shape {(B, L)}, got {output['Xhat'].shape}"
    assert output['Y'].shape == (B, L), f"Expected input shape {(B, L)}, got {output['Y'].shape}"
    
    print("✓ Forward pass successful")
    print(f"  Abundance (A): {output['A'].shape}")
    print(f"  Endmembers (E): {output['E'].shape}")
    print(f"  Reconstruction (Xhat): {output['Xhat'].shape}")
    print(f"  Input (Y): {output['Y'].shape}")
    
    # Test that abundances sum to 1 (within tolerance)
    abundance_sums = output['A'].sum(dim=1)
    expected_sum = 1.0
    tolerance = 1e-5
    assert torch.allclose(abundance_sums, torch.ones_like(abundance_sums), atol=tolerance), \
        f"Abundance sums should be 1.0, got range [{abundance_sums.min():.6f}, {abundance_sums.max():.6f}]"
    
    print(f"✓ Abundance constraint satisfied (sum ≈ 1.0)")
    
    return True

def test_spectral_dct_mamba():
    """Test the SpectralDCTMamba component specifically."""
    print("\n=== Testing SpectralDCTMamba Component ===")
    
    L, d = 20, 64
    B = 8
    
    # Test with different configurations
    for use_real_mamba in [True, False]:
        print(f"\nTesting with use_real_mamba={use_real_mamba}")
        
        spec_branch = SpectralDCTMamba(L=L, d=d, use_real_mamba=use_real_mamba)
        print(f"  Using real Mamba: {spec_branch.use_real_mamba}")
        
        # Test input
        y = torch.randn(B, L)
        
        # Forward pass
        with torch.no_grad():
            features = spec_branch(y)
        
        assert features.shape == (B, d), f"Expected output shape {(B, d)}, got {features.shape}"
        print(f"  ✓ Output shape: {features.shape}")
        
        # Check that DCT matrix is correctly sized
        assert hasattr(spec_branch, 'dct_matrix'), "DCT matrix should be registered as buffer"
        dct_shape = spec_branch.dct_matrix.shape
        expected_dct_shape = (spec_branch.n_components, L)
        assert dct_shape == expected_dct_shape, f"Expected DCT matrix shape {expected_dct_shape}, got {dct_shape}"
        print(f"  ✓ DCT matrix shape: {dct_shape}")
    
    return True

def test_comparison_with_original_backends():
    """Compare the new backend with existing ones."""
    print("\n=== Comparing Different Backends ===")
    
    L, K, patch = 16, 3, 5
    B = 4
    
    backends_to_test = ['like', 'dct', 'dct_mamba']
    if HAS_MAMBA:
        backends_to_test.append('mamba')
    
    models = {}
    outputs = {}
    
    # Create test input
    test_patch = torch.randn(B, L, patch, patch)
    test_y = torch.randn(B, L)
    
    for backend in backends_to_test:
        print(f"\nTesting backend: {backend}")
        model = DualBranchUnmixNet(L=L, K=K, backend=backend, patch=patch)
        models[backend] = model
        
        model.eval()
        with torch.no_grad():
            output = model(test_patch, test_y)
        outputs[backend] = output
        
        print(f"  Spectral: {type(model.spec).__name__}")
        print(f"  Spatial: {type(model.spa).__name__}")
        print(f"  Output shapes: A={output['A'].shape}, E={output['E'].shape}, Xhat={output['Xhat'].shape}")
    
    # Verify all backends produce consistent output shapes
    first_backend = backends_to_test[0]
    for backend in backends_to_test[1:]:
        for key in ['A', 'E', 'Xhat', 'Y']:
            shape1 = outputs[first_backend][key].shape
            shape2 = outputs[backend][key].shape
            assert shape1 == shape2, f"Shape mismatch for {key}: {backend} vs {first_backend}: {shape2} vs {shape1}"
    
    print("✓ All backends produce consistent output shapes")
    
    return True

def main():
    """Run all tests."""
    print("Testing DCT+Mamba Network Structure Implementation")
    print("=" * 60)
    
    try:
        # Test the main functionality
        success1 = test_dct_mamba_backend()
        success2 = test_spectral_dct_mamba()
        success3 = test_comparison_with_original_backends()
        
        if success1 and success2 and success3:
            print("\n" + "=" * 60)
            print("🎉 All tests passed successfully!")
            print("\nImplementation Summary:")
            print("1. ✓ Spectral branch: DCT + Mamba processing")
            print("2. ✓ Spatial branch: Real Mamba (or Mamba-like fallback)")
            print("3. ✓ New 'dct_mamba' backend option available")
            print("4. ✓ Forward pass and output shapes verified")
            print("5. ✓ Abundance constraints satisfied")
            print("6. ✓ Compatible with existing backend options")
            
            if not HAS_MAMBA:
                print("\nNote: mamba-ssm not available, using Mamba-like fallback.")
                print("Install mamba-ssm for full real Mamba functionality.")
            
            return True
        else:
            print("\n❌ Some tests failed")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)