#!/bin/bash
# 快速超参数优化演示脚本
# Quick hyperparameter optimization demo script

echo "=========================================="
echo "DPMamba 超参数优化快速演示"
echo "DPMamba Hyperparameter Optimization Quick Demo"
echo "=========================================="

# 1. 创建默认配置文件
echo "1. 创建默认配置文件 / Creating default configuration files..."
python hyperopt.py
echo ""

# 2. 创建简化的网格搜索配置进行快速演示
echo "2. 创建快速演示配置 / Creating quick demo configuration..."
cat > quick_demo_config.json << EOF
{
  "lr": [1e-4, 2e-4],
  "batch_size": [32, 64]
}
EOF

# 3. 运行快速网格搜索演示
echo "3. 网格搜索快速演示 / Quick Grid Search Demo..."
echo "运行命令 / Running command: python run_hyperopt.py --method grid --config quick_demo_config.json --dataset samson --epochs 2"
python run_hyperopt.py --method grid --config quick_demo_config.json --dataset samson --epochs 2 --results_dir ./quick_demo_results
echo ""

# 4. 显示结果
echo "4. 演示结果 / Demo Results..."
if [ -f "./quick_demo_results/best_config.json" ]; then
    echo "最佳配置 / Best Configuration:"
    cat ./quick_demo_results/best_config.json
    echo ""
    echo "重现命令 / Reproduce Command:"
    cat ./quick_demo_results/reproduce_best.sh
else
    echo "结果文件未找到 / Results file not found"
fi

echo ""
echo "=========================================="
echo "快速演示完成！/ Quick demo completed!"
echo "运行 'bash example_hyperopt.sh' 查看完整示例"
echo "Run 'bash example_hyperopt.sh' for full example"
echo "详细文档请查看 HYPEROPT_GUIDE.md"
echo "For detailed documentation, see HYPEROPT_GUIDE.md"
echo "=========================================="

# 清理演示文件
rm -f quick_demo_config.json