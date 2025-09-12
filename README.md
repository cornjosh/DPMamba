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

## 网络架构选择 / Network Architecture Options

本项目支持多种网络后端，满足不同的研究需求：

This project supports multiple network backends for different research needs:

### 后端选项 / Backend Options

- **`dct_mamba`** (推荐/Recommended): 光谱分支使用DCT+真实Mamba，空间分支使用真实Mamba / Spectral branch uses DCT+real Mamba, spatial branch uses real Mamba
- **`mamba`**: 光谱和空间分支都使用真实Mamba / Both branches use real Mamba (requires mamba-ssm)
- **`dct`**: 光谱分支使用DCT频域分析，空间分支使用Mamba-like / Spectral branch uses DCT frequency analysis, spatial branch uses Mamba-like
- **`like`**: 轻量级实现，无外部依赖 / Lightweight implementation without external dependencies

### 使用示例 / Usage Examples

```bash
# DCT+Mamba 组合 (推荐架构)
# DCT+Mamba combination (recommended architecture)
python train.py --backend dct_mamba --dataset jasper --epochs 50 --batch_size 64

# 纯Mamba架构
# Pure Mamba architecture  
python train.py --backend mamba --dataset jasper --epochs 50 --batch_size 64

# DCT频域分析
# DCT frequency analysis
python train.py --backend dct --dataset jasper --epochs 50 --batch_size 64

# 轻量级版本
# Lightweight version
python train.py --backend like --dataset jasper --epochs 50 --batch_size 64
```

## 原始训练 / Original Training

原始训练方式依然可用：

Original training method is still available:

```bash
python train.py --dataset jasper --epochs 50 --batch_size 64
```
