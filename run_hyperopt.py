#!/usr/bin/env python3
"""
Hyperparameter optimization script for DPMamba.

This script provides an easy interface to run hyperparameter optimization
with different methods (grid search, Bayesian optimization) and configurations.

Usage examples:
    # Grid search with default config
    python run_hyperopt.py --method grid --dataset jasper
    
    # Bayesian optimization with custom config
    python run_hyperopt.py --method bayes --config hyperopt_bayes_config.json --trials 100
    
    # Optuna optimization
    python run_hyperopt.py --method optuna --trials 50 --dataset urban
"""

import argparse
import sys
import os
from pathlib import Path

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import build_args
from hyperopt import (
    HyperparamConfig, GridSearchOptimizer, BayesianOptimizer, OptunaOptimizer,
    load_hyperopt_config, create_default_configs
)


def parse_hyperopt_args():
    """Parse command line arguments for hyperparameter optimization."""
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for DPMamba')
    
    # Method selection
    parser.add_argument('--method', type=str, choices=['grid', 'bayes', 'optuna'], 
                        default='grid', help='Optimization method')
    
    # Configuration
    parser.add_argument('--config', type=str, default='',
                        help='Path to hyperparameter config file (JSON/YAML)')
    parser.add_argument('--create_configs', action='store_true',
                        help='Create default configuration files and exit')
    
    # Optimization parameters
    parser.add_argument('--trials', type=int, default=50,
                        help='Number of trials for Bayesian methods')
    parser.add_argument('--metric', type=str, default='val_sad',
                        help='Metric to optimize')
    parser.add_argument('--direction', type=str, choices=['minimize', 'maximize'],
                        default='minimize', help='Optimization direction')
    
    # Output
    parser.add_argument('--results_dir', type=str, default='./hyperopt_results',
                        help='Directory to save results')
    
    # Training parameters (inherit from train.py)
    parser.add_argument('--dataset', type=str, 
                        choices=['samson','jasper','urban','apex','dc','moffett'], 
                        default='jasper', help='Dataset to use')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='Number of epochs (reduced for hyperopt)')
    parser.add_argument('--eval_batches', type=int, default=20,
                        help='Number of eval batches (reduced for hyperopt)')
    
    return parser.parse_args()


def create_base_args(hyperopt_args):
    """Create base training arguments from hyperopt arguments."""
    # Create a minimal args object with default values
    class BaseArgs:
        def __init__(self):
            # Copy relevant parameters from hyperopt_args
            self.backend = 'like'  # Use lightweight backend for faster hyperopt
            self.dataset = hyperopt_args.dataset
            self.data_dir = './data'
            self.device = hyperopt_args.device
            self.seed = hyperopt_args.seed
            
            # Training parameters (use faster settings for hyperopt)
            self.epochs = hyperopt_args.epochs
            self.batch_size = 64
            self.workers = 2  # Reduced for hyperopt
            
            # Default hyperparameters (will be overridden by optimization)
            self.lr = 2e-4
            self.weight_decay = 1e-4
            
            # Patch parameters
            self.patch = 5
            self.stride = 1
            
            # Model parameters
            self.embed_dim = 96
            self.ls = 3
            self.lp = 3
            
            # Loss weights
            self.lam_l1 = 1.0
            self.lam_sad = 0.5
            self.lam_sparse = 2e-4
            self.lam_div = 1e-2
            self.lam_e = 1e-3
            
            # Evaluation
            self.log_interval = 100  # Reduced logging for hyperopt
            self.eval_batches = hyperopt_args.eval_batches
            self.out_dir = './checkpoints'  # Will be overridden per trial
    
    return BaseArgs()


def get_default_config(method: str) -> HyperparamConfig:
    """Get default hyperparameter configuration for a method."""
    if method == 'grid':
        return HyperparamConfig(
            lr=[1e-4, 2e-4, 5e-4],
            batch_size=[32, 64],
            embed_dim=[64, 96, 128],
            ls=[2, 3, 4],
            lp=[2, 3, 4],
            lam_l1=[0.5, 1.0, 2.0],
            lam_sad=[0.1, 0.5, 1.0]
        )
    else:  # Bayesian methods
        return HyperparamConfig(
            lr=(1e-5, 1e-3),
            batch_size=(16, 128),
            embed_dim=(32, 256),
            ls=(1, 6),
            lp=(1, 6),
            weight_decay=(1e-6, 1e-2),
            lam_l1=(0.1, 5.0),
            lam_sad=(0.01, 2.0),
            lam_sparse=(1e-6, 1e-2),
            lam_div=(1e-4, 1e-1),
            lam_e=(1e-5, 1e-2)
        )


def main():
    """Main hyperparameter optimization function."""
    args = parse_hyperopt_args()
    
    # Create default configs if requested
    if args.create_configs:
        create_default_configs()
        print("Default configuration files created successfully!")
        return
    
    print(f"Starting hyperparameter optimization with {args.method} method...")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Results will be saved to: {args.results_dir}")
    
    # Load or create hyperparameter configuration
    if args.config and os.path.exists(args.config):
        print(f"Loading config from: {args.config}")
        hyperopt_config = load_hyperopt_config(args.config)
    else:
        if args.config:
            print(f"Warning: Config file {args.config} not found, using default config")
        print("Using default configuration")
        hyperopt_config = get_default_config(args.method)
    
    # Create base training arguments
    base_args = create_base_args(args)
    
    # Create optimizer based on method
    if args.method == 'grid':
        optimizer = GridSearchOptimizer(
            config=hyperopt_config,
            base_args=base_args,
            metric=args.metric,
            direction=args.direction,
            results_dir=args.results_dir
        )
    elif args.method == 'bayes':
        optimizer = BayesianOptimizer(
            config=hyperopt_config,
            base_args=base_args,
            metric=args.metric,
            direction=args.direction,
            results_dir=args.results_dir,
            n_calls=args.trials
        )
    elif args.method == 'optuna':
        optimizer = OptunaOptimizer(
            config=hyperopt_config,
            base_args=base_args,
            metric=args.metric,
            direction=args.direction,
            results_dir=args.results_dir,
            n_trials=args.trials
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    # Run optimization
    try:
        best_params, best_score = optimizer.search()
        
        print(f"\n{'='*60}")
        print("HYPERPARAMETER OPTIMIZATION COMPLETED")
        print(f"{'='*60}")
        print(f"Method: {args.method}")
        print(f"Best {args.metric}: {best_score:.6f}")
        print(f"Best parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print(f"\nFull results saved to: {args.results_dir}")
        print(f"{'='*60}")
        
        # Save command to reproduce best result
        reproduce_cmd = f"python train.py --dataset {args.dataset}"
        for key, value in best_params.items():
            reproduce_cmd += f" --{key} {value}"
        
        with open(Path(args.results_dir) / 'reproduce_best.sh', 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Command to reproduce the best hyperparameter configuration\n")
            f.write(f"{reproduce_cmd}\n")
        
        print(f"To reproduce the best result, run:")
        print(f"  {reproduce_cmd}")
        
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
        optimizer.save_summary()
    except Exception as e:
        print(f"\nError during optimization: {e}")
        if hasattr(optimizer, 'save_summary'):
            optimizer.save_summary()
        raise


if __name__ == '__main__':
    main()