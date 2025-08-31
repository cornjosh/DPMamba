# 超参数搜索与优化使用指南

本项目集成了网格搜索与贝叶斯优化的超参数搜索方案，支持自动寻找DPMamba模型的最优超参数配置。

## 功能特性

- **多种优化方法**: 支持网格搜索、scikit-optimize贝叶斯优化、Optuna优化
- **灵活配置**: 支持JSON/YAML配置文件自定义搜索空间
- **全面的超参数**: 支持学习率、批大小、模型结构、损失权重等参数优化
- **结果保存**: 自动保存优化过程和最佳配置
- **可重现性**: 提供最佳配置的重现命令

## 安装依赖

```bash
pip install scikit-optimize optuna
```

## 快速开始

### 1. 创建默认配置文件

```bash
python hyperopt.py
```

这将创建两个默认配置文件：
- `hyperopt_grid_config.json`: 网格搜索配置
- `hyperopt_bayes_config.json`: 贝叶斯优化配置

### 2. 运行超参数优化

#### 网格搜索
```bash
# 使用默认网格搜索配置
python run_hyperopt.py --method grid --dataset jasper

# 使用自定义配置文件
python run_hyperopt.py --method grid --config hyperopt_grid_config.json --dataset urban
```

#### 贝叶斯优化 (scikit-optimize)
```bash
# 运行50次贝叶斯优化试验
python run_hyperopt.py --method bayes --trials 50 --dataset jasper

# 使用自定义配置
python run_hyperopt.py --method bayes --config hyperopt_bayes_config.json --trials 100
```

#### Optuna优化
```bash
# 运行Optuna优化
python run_hyperopt.py --method optuna --trials 50 --dataset jasper
```

## 详细使用说明

### 命令行参数

#### 基本参数
- `--method`: 优化方法 (`grid`, `bayes`, `optuna`)
- `--config`: 配置文件路径 (JSON/YAML格式)
- `--trials`: 贝叶斯方法的试验次数
- `--dataset`: 数据集名称
- `--device`: 计算设备 (cuda:0, cpu等)

#### 优化设置
- `--metric`: 优化指标 (默认: val_sad)
- `--direction`: 优化方向 (`minimize`, `maximize`)
- `--results_dir`: 结果保存目录

### 配置文件格式

#### 网格搜索配置 (hyperopt_grid_config.json)
```json
{
  "lr": [1e-4, 2e-4, 5e-4],
  "batch_size": [32, 64, 128],
  "embed_dim": [64, 96, 128],
  "ls": [2, 3, 4],
  "lp": [2, 3, 4],
  "lam_l1": [0.5, 1.0, 2.0],
  "lam_sad": [0.1, 0.5, 1.0]
}
```

#### 贝叶斯优化配置 (hyperopt_bayes_config.json)
```json
{
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
```

### 支持的超参数

#### 训练参数
- `lr`: 学习率
- `batch_size`: 批大小
- `weight_decay`: 权重衰减

#### 模型结构
- `embed_dim`: 嵌入维度
- `ls`: 光谱分支层数
- `lp`: 空间分支层数
- `patch`: 补丁大小

#### 损失权重
- `lam_l1`: L1损失权重
- `lam_sad`: SAD损失权重
- `lam_sparse`: 稀疏损失权重
- `lam_div`: 端元多样性损失权重
- `lam_e`: 端元损失权重

## 结果分析

优化完成后，结果保存在指定目录中：

```
hyperopt_results/
├── best_config.json          # 最佳配置
├── summary.json              # 优化总结
├── trial_0001.json           # 各试验结果
├── trial_0002.json
├── ...
└── reproduce_best.sh         # 重现最佳结果的命令
```

### 重现最佳结果

```bash
# 运行生成的重现脚本
bash hyperopt_results/reproduce_best.sh

# 或者手动运行最佳配置
python train.py --dataset jasper --lr 0.0002 --batch_size 64 --embed_dim 96 ...
```

## 高级用法

### 1. 自定义搜索空间

创建自定义配置文件 `my_config.json`:
```json
{
  "lr": [1e-4, 5e-4, 1e-3],
  "embed_dim": [64, 128, 256],
  "lam_l1": [0.1, 1.0, 10.0]
}
```

运行优化：
```bash
python run_hyperopt.py --config my_config.json --method grid
```

### 2. 多数据集优化

```bash
# 不同数据集的优化
for dataset in jasper urban samson; do
    python run_hyperopt.py --method bayes --dataset $dataset --trials 30 \
        --results_dir ./hyperopt_results_$dataset
done
```

### 3. 继续中断的优化

Optuna方法支持断点续传，可以在中断后继续优化。

### 4. 自定义优化指标

如果需要优化其他指标，可以修改 `train.py` 中的返回值，例如返回验证RMSE：

```python
# 在train函数末尾
val_sad, val_rmse = evaluate_epoch(model, train_loader, device, max_batches=args.eval_batches)
return val_rmse  # 返回RMSE而不是SAD
```

## 性能优化建议

1. **减少epochs**: 超参数优化时使用较少的epochs（如20）以加快速度
2. **使用GPU**: 确保使用GPU加速训练
3. **并行优化**: 在多GPU环境下可以并行运行多个试验
4. **合理选择搜索空间**: 避免过大的搜索空间导致优化效率低下

## 常见问题

### Q: 优化过程中出现内存不足怎么办？
A: 可以减少batch_size的搜索范围，或使用更少的workers。

### Q: 如何加快优化速度？
A: 减少epochs数量，使用较小的eval_batches，选择合适的搜索空间大小。

### Q: 网格搜索和贝叶斯优化如何选择？
A: 网格搜索适合参数较少的情况，贝叶斯优化适合参数较多且试验成本较高的情况。

### Q: 如何设置合理的搜索范围？
A: 基于现有经验和文献，设置包含当前最佳值的合理范围，避免过大范围导致效率低下。

## 示例命令汇总

```bash
# 创建配置文件
python hyperopt.py

# 快速网格搜索
python run_hyperopt.py --method grid --dataset jasper --epochs 10

# 详细贝叶斯优化
python run_hyperopt.py --method bayes --trials 100 --dataset urban \
    --config hyperopt_bayes_config.json --epochs 20

# Optuna优化
python run_hyperopt.py --method optuna --trials 50 --dataset samson

# 自定义结果目录
python run_hyperopt.py --method grid --results_dir ./my_hyperopt_results
```