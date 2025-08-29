# DPMamba
Dual - perspective mamba unmixing network

## 新增功能 / New Features

### 超参数搜索与优化 / Hyperparameter Search & Optimization

本项目现已集成网格搜索与贝叶斯优化的超参数搜索方案，支持自动寻找最优模型配置。

This project now includes integrated grid search and Bayesian optimization for hyperparameter tuning, supporting automatic discovery of optimal model configurations.

#### 快速开始 / Quick Start

```bash
# 创建默认配置文件 / Create default configuration files
python hyperopt.py

# 网格搜索 / Grid Search
python run_hyperopt.py --method grid --dataset jasper

# 贝叶斯优化 / Bayesian Optimization
python run_hyperopt.py --method bayes --trials 50 --dataset jasper

# Optuna优化 / Optuna Optimization
python run_hyperopt.py --method optuna --trials 50 --dataset jasper
```

#### 详细文档 / Detailed Documentation

请查看 [HYPEROPT_GUIDE.md](HYPEROPT_GUIDE.md) 获取完整的使用指南。

See [HYPEROPT_GUIDE.md](HYPEROPT_GUIDE.md) for complete usage guide.

#### 示例 / Example

快速演示：

Quick demo:

```bash
bash quick_demo.sh
```

完整示例：

Full example:

```bash
bash example_hyperopt.sh
```

## 原始训练 / Original Training

原始训练方式依然可用：

Original training method is still available:

```bash
python train.py --dataset jasper --epochs 50 --batch_size 64
```
