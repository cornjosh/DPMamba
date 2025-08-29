#!/usr/bin/env python3
"""
Test script for hyperparameter optimization functionality.
This runs a quick test with minimal parameters to verify the implementation works.
"""

import sys
import os
import json
import tempfile
import shutil
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hyperopt import HyperparamConfig, GridSearchOptimizer
from run_hyperopt import create_base_args


def create_test_config():
    """Create a minimal test configuration."""
    return HyperparamConfig(
        lr=[1e-4, 2e-4],  # Just 2 values for quick test
        batch_size=[32, 64]  # Just 2 values for quick test
    )


def test_hyperopt():
    """Test hyperparameter optimization with minimal settings."""
    print("Testing hyperparameter optimization...")
    
    # Create test arguments
    class TestArgs:
        def __init__(self):
            self.dataset = 'samson'  # Use samson dataset (smallest)
            self.device = 'cpu'  # Use CPU for test
            self.seed = 1234
            self.epochs = 2  # Very few epochs for quick test
            self.eval_batches = 5  # Very few eval batches
    
    test_args = TestArgs()
    base_args = create_base_args(test_args)
    
    # Override with minimal settings for testing
    base_args.epochs = 2
    base_args.eval_batches = 5
    base_args.log_interval = 10
    base_args.workers = 0  # No multiprocessing for test
    
    # Create test configuration
    config = create_test_config()
    
    # Create temporary results directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Create optimizer
        optimizer = GridSearchOptimizer(
            config=config,
            base_args=base_args,
            metric='val_sad',
            direction='minimize',
            results_dir=temp_dir
        )
        
        try:
            # Run optimization (should test 4 combinations: 2x2)
            print("Running optimization...")
            best_params, best_score = optimizer.search()
            
            print(f"Optimization completed successfully!")
            print(f"Best score: {best_score}")
            print(f"Best params: {best_params}")
            
            # Check that results were saved
            results_dir = Path(temp_dir)
            assert (results_dir / 'best_config.json').exists(), "best_config.json not found"
            assert (results_dir / 'summary.json').exists(), "summary.json not found"
            
            # Load and verify results
            with open(results_dir / 'best_config.json', 'r') as f:
                best_config = json.load(f)
            
            assert 'best_score' in best_config, "best_score not in results"
            assert 'best_params' in best_config, "best_params not in results"
            
            print("All tests passed!")
            return True
            
        except Exception as e:
            print(f"Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == '__main__':
    success = test_hyperopt()
    sys.exit(0 if success else 1)