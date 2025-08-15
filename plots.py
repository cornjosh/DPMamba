import numpy as np
from matplotlib import pyplot as plt

# 绘制丰度图
def plot_abundance(ground_truth, estimated, abu_spa, abu_spr, em, save_dir):
    # 创建一个新的图像，大小为12x12英寸，分辨率为150dpi
    plt.figure(figsize=(12, 12), dpi=150)
    for i in range(em):
        # 绘制真实丰度图
        plt.subplot(4, em, i + 1)
        plt.imshow(ground_truth[:, :, i], cmap='jet')
        if i == 0:
            plt.ylabel("GT")

    for i in range(em):
        # 绘制spa分支丰度图
        plt.subplot(4, em, em + i + 1)
        plt.imshow(abu_spa[:, :, i], cmap='jet')
        if i == 0:
            plt.ylabel("spa")

    for i in range(em):
        # 绘制spr分支丰度图
        plt.subplot(4, em, 2 * em + i + 1)
        plt.imshow(abu_spr[:, :, i], cmap='jet')
        if i == 0:
            plt.ylabel("spr")

    for i in range(em):
        # 绘制估计丰度图
        plt.subplot(4, em, 3 * em + i + 1)
        plt.imshow(estimated[:, :, i], cmap='jet')
        if i == 0:
            plt.ylabel("fused")

    plt.tight_layout()

    # 保存图像到指定目录
    plt.savefig(save_dir + "abundance.png")

# 绘制端元图
def plot_endmembers(target, pred, endmem_spa, endmem_spr, em, save_dir):
    # 创建一个新的图像，大小为12x6英寸，分辨率为150dpi
    plt.figure(figsize=(12, 6), dpi=150)
    for i in range(em):
        # 绘制端元曲线
        plt.subplot(2, em // 2 if em % 2 == 0 else em, i + 1)
        plt.plot(target[:, i], label="GT")
        plt.plot(endmem_spa[:, i], label="spa")
        plt.plot(endmem_spr[:, i], label="spr")
        plt.plot(pred[:, i], label="fused")
        plt.legend(loc="upper left")
    plt.tight_layout()

    # 保存图像到指定目录
    plt.savefig(save_dir + "end_members.png")
