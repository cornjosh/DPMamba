"""
Hyperparameter optimization module for DPMamba.
Supports both grid search and Bayesian optimization using scikit-optimize and Optuna.

This module provides a unified interface for hyperparameter search across the DPMamba model,
supporting various optimization strategies and parameter types.
"""

import os
import json
import yaml
import itertools
from typing import Dict, List, Any, Tuple, Union, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import ParameterGrid
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import optuna

from train import train, build_args, set_seed


@dataclass
class HyperparamConfig:
    """Configuration for hyperparameter search spaces."""
    
    # Optimization parameters
    lr: Union[List[float], Tuple[float, float]] = None
    batch_size: Union[List[int], Tuple[int, int]] = None
    embed_dim: Union[List[int], Tuple[int, int]] = None
    ls: Union[List[int], Tuple[int, int]] = None  # spectral layers
    lp: Union[List[int], Tuple[int, int]] = None  # spatial layers
    weight_decay: Union[List[float], Tuple[float, float]] = None
    patch: Union[List[int], Tuple[int, int]] = None
    
    # Loss weights
    lam_l1: Union[List[float], Tuple[float, float]] = None
    lam_sad: Union[List[float], Tuple[float, float]] = None
    lam_sparse: Union[List[float], Tuple[float, float]] = None
    lam_div: Union[List[float], Tuple[float, float]] = None
    lam_e: Union[List[float], Tuple[float, float]] = None
    
    # Categorical parameters
    backend: List[str] = None


class HyperparameterSearcher:
    """Base class for hyperparameter optimization."""
    
    def __init__(self, 
                 config: HyperparamConfig,
                 base_args: Any,
                 metric: str = 'val_sad',
                 direction: str = 'minimize',
                 results_dir: str = './hyperopt_results'):
        """
        Initialize hyperparameter searcher.
        
        Args:
            config: Hyperparameter configuration
            base_args: Base training arguments
            metric: Metric to optimize ('val_sad', 'val_loss', etc.)
            direction: 'minimize' or 'maximize'
            results_dir: Directory to save results
        """
        self.config = config
        self.base_args = base_args
        self.metric = metric
        self.direction = direction
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.best_score = float('inf') if direction == 'minimize' else float('-inf')
        self.best_params = None
        self.trial_results = []
        
    def _create_args_from_params(self, params: Dict[str, Any]) -> Any:
        """Create training arguments from hyperparameter values."""
        import copy
        args = copy.deepcopy(self.base_args)
        
        # Update args with hyperparameters
        for key, value in params.items():
            if hasattr(args, key):
                setattr(args, key, value)
        
        return args
    
    def _evaluate_params(self, params: Dict[str, Any]) -> float:
        """Evaluate a set of hyperparameters."""
        try:
            # Set seed for reproducibility
            set_seed(self.base_args.seed)
            
            # Create args with hyperparameters
            args = self._create_args_from_params(params)
            
            # Create unique output directory for this trial
            trial_id = len(self.trial_results) + 1
            args.out_dir = str(self.results_dir / f'trial_{trial_id:04d}')
            
            # Train model and get validation score
            val_score = train(args)
            
            # Record result
            result = {
                'trial_id': trial_id,
                'params': params,
                'score': val_score,
                'metric': self.metric
            }
            self.trial_results.append(result)
            
            # Update best score
            is_better = (val_score < self.best_score if self.direction == 'minimize' 
                        else val_score > self.best_score)
            if is_better:
                self.best_score = val_score
                self.best_params = params.copy()
                
                # Save best configuration
                self._save_best_config()
            
            # Save trial result
            self._save_trial_result(result)
            
            print(f"Trial {trial_id}: {self.metric}={val_score:.6f}, params={params}")
            
            return val_score
            
        except Exception as e:
            print(f"Error in trial: {e}")
            # Return worst possible score for failed trials
            return float('inf') if self.direction == 'minimize' else float('-inf')
    
    def _save_best_config(self):
        """Save the best configuration found so far."""
        best_config = {
            'best_score': self.best_score,
            'best_params': self.best_params,
            'metric': self.metric,
            'direction': self.direction
        }
        
        with open(self.results_dir / 'best_config.json', 'w') as f:
            json.dump(best_config, f, indent=2)
    
    def _save_trial_result(self, result: Dict[str, Any]):
        """Save individual trial result."""
        trial_file = self.results_dir / f"trial_{result['trial_id']:04d}.json"
        with open(trial_file, 'w') as f:
            json.dump(result, f, indent=2)
    
    def save_summary(self):
        """Save summary of all trials."""
        summary = {
            'total_trials': len(self.trial_results),
            'best_score': self.best_score,
            'best_params': self.best_params,
            'metric': self.metric,
            'direction': self.direction,
            'all_results': self.trial_results
        }
        
        with open(self.results_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nHyperparameter search completed!")
        print(f"Total trials: {len(self.trial_results)}")
        print(f"Best {self.metric}: {self.best_score:.6f}")
        print(f"Best parameters: {self.best_params}")
        print(f"Results saved to: {self.results_dir}")


class GridSearchOptimizer(HyperparameterSearcher):
    """Grid search hyperparameter optimizer."""
    
    def search(self) -> Tuple[Dict[str, Any], float]:
        """Run grid search optimization."""
        print("Starting Grid Search optimization...")
        
        # Convert config to parameter grid format
        param_grid = {}
        for key, value in asdict(self.config).items():
            if value is not None:
                if isinstance(value, (list, tuple)):
                    param_grid[key] = value
                else:
                    param_grid[key] = [value]
        
        # Generate all parameter combinations
        param_combinations = list(ParameterGrid(param_grid))
        total_trials = len(param_combinations)
        
        print(f"Total combinations to evaluate: {total_trials}")
        
        # Evaluate each combination
        for i, params in enumerate(param_combinations, 1):
            print(f"\nEvaluating combination {i}/{total_trials}")
            self._evaluate_params(params)
        
        self.save_summary()
        return self.best_params, self.best_score


class BayesianOptimizer(HyperparameterSearcher):
    """Bayesian optimization using scikit-optimize."""
    
    def __init__(self, *args, n_calls: int = 50, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_calls = n_calls
        
    def _create_search_space(self) -> List[Any]:
        """Create search space for scikit-optimize."""
        space = []
        param_names = []
        
        for key, value in asdict(self.config).items():
            if value is not None:
                if key in ['lr', 'weight_decay', 'lam_l1', 'lam_sad', 'lam_sparse', 'lam_div', 'lam_e']:
                    # Continuous parameters
                    if isinstance(value, (list, tuple)) and len(value) == 2:
                        space.append(Real(value[0], value[1], name=key))
                        param_names.append(key)
                elif key in ['batch_size', 'embed_dim', 'ls', 'lp', 'patch']:
                    # Integer parameters
                    if isinstance(value, (list, tuple)) and len(value) == 2:
                        space.append(Integer(value[0], value[1], name=key))
                        param_names.append(key)
                elif key == 'backend':
                    # Categorical parameters
                    space.append(Categorical(value, name=key))
                    param_names.append(key)
        
        return space, param_names
    
    def search(self) -> Tuple[Dict[str, Any], float]:
        """Run Bayesian optimization."""
        print("Starting Bayesian optimization...")
        
        space, param_names = self._create_search_space()
        
        @use_named_args(space)
        def objective(**params):
            return self._evaluate_params(params)
        
        print(f"Running {self.n_calls} trials with Bayesian optimization...")
        
        # Run optimization
        result = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=self.n_calls,
            random_state=self.base_args.seed,
            acq_func='EI'  # Expected Improvement
        )
        
        # Extract best parameters
        best_params = {name: value for name, value in zip(param_names, result.x)}
        
        self.save_summary()
        return best_params, result.fun


class OptunaOptimizer(HyperparameterSearcher):
    """Bayesian optimization using Optuna."""
    
    def __init__(self, *args, n_trials: int = 50, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_trials = n_trials
        
    def _create_objective(self, trial):
        """Create objective function for Optuna."""
        params = {}
        
        for key, value in asdict(self.config).items():
            if value is not None:
                if key in ['lr', 'weight_decay', 'lam_l1', 'lam_sad', 'lam_sparse', 'lam_div', 'lam_e']:
                    # Continuous parameters
                    if isinstance(value, (list, tuple)) and len(value) == 2:
                        params[key] = trial.suggest_float(key, value[0], value[1])
                elif key in ['batch_size', 'embed_dim', 'ls', 'lp', 'patch']:
                    # Integer parameters
                    if isinstance(value, (list, tuple)) and len(value) == 2:
                        params[key] = trial.suggest_int(key, value[0], value[1])
                elif key == 'backend':
                    # Categorical parameters
                    params[key] = trial.suggest_categorical(key, value)
        
        return self._evaluate_params(params)
    
    def search(self) -> Tuple[Dict[str, Any], float]:
        """Run Optuna optimization."""
        print("Starting Optuna optimization...")
        
        # Create study
        direction = 'minimize' if self.direction == 'minimize' else 'maximize'
        study = optuna.create_study(direction=direction)
        
        print(f"Running {self.n_trials} trials with Optuna...")
        
        # Run optimization
        study.optimize(self._create_objective, n_trials=self.n_trials)
        
        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        # Update best results
        self.best_params = best_params
        self.best_score = best_score
        
        self.save_summary()
        return best_params, best_score


def load_hyperopt_config(config_path: str) -> HyperparamConfig:
    """Load hyperparameter configuration from file."""
    config_path = Path(config_path)
    
    if config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    elif config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    return HyperparamConfig(**config_dict)


def create_default_configs():
    """Create default configuration files for hyperparameter search."""
    
    # Grid search config
    grid_config = {
        "lr": [1e-4, 2e-4, 5e-4],
        "batch_size": [32, 64, 128],
        "embed_dim": [64, 96, 128],
        "ls": [2, 3, 4],
        "lp": [2, 3, 4],
        "lam_l1": [0.5, 1.0, 2.0],
        "lam_sad": [0.1, 0.5, 1.0]
    }
    
    # Bayesian optimization config  
    bayes_config = {
        "lr": [1e-5, 1e-3],
        "batch_size": [16, 256],
        "embed_dim": [32, 256],
        "ls": [1, 6],
        "lp": [1, 6],
        "weight_decay": [1e-6, 1e-2],
        "lam_l1": [0.1, 5.0],
        "lam_sad": [0.01, 2.0],
        "lam_sparse": [1e-6, 1e-2],
        "lam_div": [1e-4, 1e-1],
        "lam_e": [1e-5, 1e-2]
    }
    
    # Save configs
    with open('hyperopt_grid_config.json', 'w') as f:
        json.dump(grid_config, f, indent=2)
    
    with open('hyperopt_bayes_config.json', 'w') as f:
        json.dump(bayes_config, f, indent=2)
    
    print("Created default configuration files:")
    print("- hyperopt_grid_config.json (for grid search)")
    print("- hyperopt_bayes_config.json (for Bayesian optimization)")


if __name__ == "__main__":
    # Create default configuration files when run directly
    create_default_configs()