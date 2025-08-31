#!/bin/bash
# 超参数优化示例脚本
# Hyperparameter optimization example script

echo "=========================================="
echo "DPMamba 超参数优化示例"
echo "DPMamba Hyperparameter Optimization Example"
echo "=========================================="

# 1. 创建默认配置文件
echo "1. 创建默认配置文件 / Creating default configuration files..."
python hyperopt.py
echo ""

# 2. 网格搜索示例
echo "2. 网格搜索示例 / Grid Search Example..."
echo "运行命令 / Running command: python run_hyperopt.py --method grid --dataset samson --epochs 5 --trials 4"
python run_hyperopt.py --method grid --dataset samson --epochs 5 --results_dir ./example_grid_results
echo ""

# 3. 贝叶斯优化示例
echo "3. 贝叶斯优化示例 / Bayesian Optimization Example..."
echo "运行命令 / Running command: python run_hyperopt.py --method bayes --dataset samson --epochs 5 --trials 8"
python run_hyperopt.py --method bayes --dataset samson --epochs 5 --trials 8 --results_dir ./example_bayes_results
echo ""

# 4. 显示结果
echo "4. 优化结果 / Optimization Results..."
echo ""
echo "网格搜索结果 / Grid Search Results:"
if [ -f "./example_grid_results/best_config.json" ]; then
    cat ./example_grid_results/best_config.json
    echo ""
    echo "重现命令 / Reproduce command:"
    cat ./example_grid_results/reproduce_best.sh
else
    echo "结果文件未找到 / Results file not found"
fi

echo ""
echo "贝叶斯优化结果 / Bayesian Optimization Results:"
if [ -f "./example_bayes_results/best_config.json" ]; then
    cat ./example_bayes_results/best_config.json
    echo ""
    echo "重现命令 / Reproduce command:"
    cat ./example_bayes_results/reproduce_best.sh
else
    echo "结果文件未找到 / Results file not found"
fi

echo ""
echo "=========================================="
echo "示例完成！/ Example completed!"
echo "详细文档请查看 HYPEROPT_GUIDE.md"
echo "For detailed documentation, see HYPEROPT_GUIDE.md"
echo "=========================================="