import os
import numpy as np
from matplotlib import pyplot as plt

# 绘制丰度图
def plot_abundance(ground_truth, estimated, abu_spa, abu_spr, em, save_dir):
    # 输入自适应：支持 [H,W,em] / [N,em] / [em,N]（当 N 为完全平方数时）
    def _to_hwk(arr, em):
        a = np.asarray(arr)
        if a.ndim == 3:
            return a
        if a.ndim == 2:
            # 形状 [N, em]
            if a.shape[1] == em:
                N = a.shape[0]
                s = int(np.round(np.sqrt(N)))
                if s * s == N:
                    return a.reshape(s, s, em)
            # 形状 [em, N] -> 转置后尝试
            if a.shape[0] == em:
                b = a.T
                N = b.shape[0]
                s = int(np.round(np.sqrt(N)))
                if s * s == N:
                    return b.reshape(s, s, em)
        raise ValueError(f"无法将数组转换为 HxWx{em} 形状，当前形状 {a.shape}")

    # 统一转换输入
    gt = _to_hwk(ground_truth, em)
    est = _to_hwk(estimated, em)
    spa = _to_hwk(abu_spa, em)
    spr = _to_hwk(abu_spr, em)

    # 创建一个新的图像，大小为12x12英寸，分辨率为150dpi
    plt.figure(figsize=(12, 12), dpi=150)
    for i in range(em):
        # 绘制真实丰度图
        plt.subplot(4, em, i + 1)
        plt.imshow(gt[:, :, i], cmap='jet')
        if i == 0:
            plt.ylabel("GT")

    for i in range(em):
        # 绘制spa分支丰度图
        plt.subplot(4, em, em + i + 1)
        plt.imshow(spa[:, :, i], cmap='jet')
        if i == 0:
            plt.ylabel("spa")

    for i in range(em):
        # 绘制spr分支丰度图
        plt.subplot(4, em, 2 * em + i + 1)
        plt.imshow(spr[:, :, i], cmap='jet')
        if i == 0:
            plt.ylabel("spr")

    for i in range(em):
        # 绘制估计丰度图
        plt.subplot(4, em, 3 * em + i + 1)
        plt.imshow(est[:, :, i], cmap='jet')
        if i == 0:
            plt.ylabel("fused")

    plt.tight_layout()

    # 保存图像到指定目录（确保目录存在）
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "abundance.png"))

# 绘制端元图
def plot_endmembers(target, pred, endmem_spa, endmem_spr, em, save_dir):
    # 创建一个新的图像，大小为12x6英寸，分辨率为150dpi
    # 输入自适应：支持 [L,em] 或 [em,L]
    def _to_lk(arr, em):
        a = np.asarray(arr)
        if a.ndim == 2:
            # 期待 [L, em]
            if a.shape[1] == em:
                return a
            # 可能是 [em, L]，转置
            if a.shape[0] == em:
                return a.T
        raise ValueError(f"无法将数组转换为 Lx{em} 形状，当前形状 {a.shape}")

    tgt = _to_lk(target, em)
    prd = _to_lk(pred, em)
    ssp = _to_lk(endmem_spa, em)
    spr = _to_lk(endmem_spr, em)

    plt.figure(figsize=(12, 6), dpi=150)
    cols = em // 2 if em % 2 == 0 else em
    for i in range(em):
        # 绘制端元曲线
        plt.subplot(2, cols, i + 1)
        plt.plot(tgt[:, i], label="GT")
        plt.plot(ssp[:, i], label="spa")
        plt.plot(spr[:, i], label="spr")
        plt.plot(prd[:, i], label="fused")
        plt.legend(loc="upper left")
    plt.tight_layout()

    # 保存图像到指定目录（确保目录存在）
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "end_members.png"))
