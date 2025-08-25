#!/usr/bin/env bash
set -e

# Baseline（原方法）
python train.py \
  --dataset jasper \
  --data_dir ./data \
  --epochs 50 --batch_size 64 --patch 5 --stride 1 \
  --lr 2e-4 --device cuda:0

# Unsupervised-Improved（仅用 Y 训练）
python train.py \
  --dataset jasper \
  --data_dir ./data \
  --epochs 100 --batch_size 64 --patch 5 --stride 1 \
  --lr 2e-4 --weight_decay 1e-4 --device cuda:0 \
  --lam_div 1e-3 --lam_vol 5e-4 --tau 0.8
# 如接受 M1 作为"来自 Y 的先验"，可追加： --lam_esam 0.3