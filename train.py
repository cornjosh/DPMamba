"""
Dual-branch (Spectral/Spatial) Mamba-like Autoencoder for Hyperspectral Unmixing
Dataset: Jasper Ridge (L=198, K=4, H=W=100 assumed by provided loader)

- 数据载入：沿用你的 Data 类（datasets.py）
- PatchDataset：基于 Data.get('hs_img') 重建 H×W×L 立方体，再按滑窗提供 patch
- 模型：光谱分支(1D Mamba-like) + 空间分支(2D Mamba-like) + 三阶段融合 + 可解释解码头(A@E)
- 损失：L1 + SAD + 稀疏(ℓ0.5) + 端元多样性；（TV 可选，默认关闭）
- 说明：本实现支持 `--backend mamba`（需 `pip install mamba-ssm`）与 `--backend like`（内置轻量近似）。若你已装 VSSM 也可按同形状替换。

运行示例：
    python jasper_dual_mamba_unmix.py \
        --dataset jasper \
        --data_dir ./data \
        --epochs 50 --batch_size 64 --patch 5 --stride 1 \
        --lr 2e-4 --device cuda:0

需要的数据文件：
    ./data/jasper_dataset.mat
文件格式与字段需与 datasets.py 一致（Y, A, M, M1）。
"""
from __future__ import annotations

import os
import math
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==== Optional real Mamba backend (install: pip install mamba-ssm) ====
try:
    from mamba_ssm import Mamba as MambaLayer  # expects inputs [B, L, d]
    HAS_MAMBA = True
except Exception:
    HAS_MAMBA = False

# ==== 1) 数据载入：使用你的 Data 类 ====
from datasets import Data  # 确保 datasets.py 与本文件同级

# ==== 2) 工具函数 ====

def set_seed(seed: int = 1234):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tensor_sad(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """光谱角 (弧度)，逐样本，输入形状 [B, L]."""
    x_norm = x / (x.norm(dim=-1, keepdim=True) + eps)
    y_norm = y / (y.norm(dim=-1, keepdim=True) + eps)
    cos = (x_norm * y_norm).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.acos(cos)


def endmember_diversity_loss(E: torch.Tensor) -> torch.Tensor:
    """端元多样性：pairwise cos^2(Ek, Ej) 求和，E:[K, L]."""
    E = F.normalize(E, dim=-1)
    K = E.size(0)
    sim = E @ E.t()  # [K,K]
    mask = ~torch.eye(K, dtype=torch.bool, device=E.device)
    return (sim[mask] ** 2).mean()


def l05_sparsity(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """ℓ0.5 稀疏性（可作为 A 的稀疏正则）；A>=0 时有效。"""
    return torch.sqrt(A + eps).sum(dim=-1).mean()


def rmse(x: torch.Tensor, y: torch.Tensor) -> float:
    return torch.sqrt(F.mse_loss(x, y)).item()


# ==== 3) PatchDataset：从 Data 重建立方体并提取滑窗 ====
class PatchDataset(Dataset):
    def __init__(self,
                 data_obj: Data,
                 patch: int = 5,
                 stride: int = 1,
                 use_center_abd: bool = True):
        super().__init__()
        self.data = data_obj
        self.patch = patch
        self.stride = stride
        self.use_center_abd = use_center_abd
        
        Y = self.data.get("hs_img")  # [N, L] float
        A = self.data.get("abd_map")  # [N, K] float
        L = self.data.get_L()
        col = self.data.get_col()  # 假定 H=W=col
        H = W = col
        assert Y.dim() == 2 and Y.size(1) == L, "Y 应为 [N, L]"
        assert Y.size(0) == H * W, f"N={Y.size(0)} 与 H*W={H*W} 不一致"

        self.H, self.W, self.L = H, W, L
        self.K = self.data.get_P()

        # 重建立方体并缓存到 CPU（节省显存；按需搬运到 device）
        self.cube = Y.view(H, W, L).cpu()  # [H,W,L]
        self.abd_full = A.view(H, W, self.K).cpu()  # [H,W,K]

        r = patch // 2
        centers = []
        for i in range(r, H - r, stride):
            for j in range(r, W - r, stride):
                centers.append((i, j))
        self.centers = centers

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, idx: int):
        i, j = self.centers[idx]
        r = self.patch // 2
        # 取 patch: [p, p, L]
        patch_ = self.cube[i - r:i + r + 1, j - r:j + r + 1, :]  # [p,p,L]
        # 中心像素的光谱与丰度
        y_ctr = self.cube[i, j, :]          # [L]
        a_ctr = self.abd_full[i, j, :]      # [K]
        # 调整为张量形状
        patch_ = patch_.permute(2, 0, 1).contiguous()  # [L, p, p]
        return patch_, y_ctr, a_ctr


# ==== 4) Mamba / VSSM Backends ====
class GLU(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(d, 2 * d)
    def forward(self, x):
        u, v = self.proj(x).chunk(2, dim=-1)
        return u * torch.sigmoid(v)

# ---- Real Mamba backend (if available) ----
class MambaResBlock(nn.Module):
    def __init__(self, d: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if not HAS_MAMBA:
            raise RuntimeError("mamba-ssm is not installed. Please `pip install mamba-ssm`. Or use --backend like")
        self.norm = nn.LayerNorm(d)
        self.mamba = MambaLayer(d_model=d, d_state=d_state, d_conv=d_conv, expand=expand)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d]
        return x + self.mamba(self.norm(x))

class SpectralMambaReal(nn.Module):
    """Use real Mamba to model per-pixel spectrum sequence (length=L). Output [B,d]."""
    def __init__(self, L: int, d: int = 96, layers: int = 3):
        super().__init__()
        self.in_proj = nn.Linear(1, d)  # 1 -> d per band
        self.blocks = nn.ModuleList([MambaResBlock(d) for _ in range(layers)])
        self.out_norm = nn.LayerNorm(d)
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: [B, L]
        z = self.in_proj(y.unsqueeze(-1))    # [B,L,d]
        for blk in self.blocks:
            z = blk(z)
        z = self.out_norm(z).mean(dim=1)     # global average over bands -> [B,d]
        return z

class SpatialMambaReal(nn.Module):
    """Use real Mamba over flattened spatial tokens of a patch. Input [B,L,p,p] -> [B,d]."""
    def __init__(self, L: int, d: int = 96, layers: int = 3, patch: int = 5):
        super().__init__()
        self.in_proj = nn.Conv2d(L, d, kernel_size=1)
        self.blocks = nn.ModuleList([MambaResBlock(d) for _ in range(layers)])
        self.out_norm = nn.LayerNorm(d)
        self.patch = patch
    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(patch)                 # [B,d,p,p]
        z = z.flatten(2).transpose(1, 2)        # [B, p*p, d]
        for blk in self.blocks:
            z = blk(z)
        z = self.out_norm(z).mean(dim=1)        # [B,d]
        return z

# ---- Lightweight fallback (no external deps) ----
class SpectralMambaLike(nn.Module):
    """1D Mamba-like：Depthwise Conv1d + GLU + 残差，输入 [B, L]，内部转 [B, d, L]."""
    def __init__(self, L: int, d: int = 96, layers: int = 3, kernel_size: int = 5):
        super().__init__()
        self.in_proj = nn.Conv1d(1, d, kernel_size=1)
        blks = []
        for _ in range(layers):
            blks += [
                nn.Conv1d(d, d, kernel_size, padding=kernel_size // 2, groups=d),  # depthwise
                nn.Conv1d(d, d, kernel_size=1),
                nn.GELU(),
            ]
        self.blocks = nn.Sequential(*blks)
        self.out_proj = nn.Conv1d(d, d, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool1d(1)  # 全局压缩到 [B, d, 1]
        self.norm = nn.LayerNorm(d)
        self.glu = GLU(d)
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        x = y.unsqueeze(1)  # [B,1,L]
        z = self.in_proj(x)  # [B,d,L]
        z = z + self.blocks(z)
        z = self.out_proj(z)
        z = self.pool(z).squeeze(-1)  # [B,d]
        z = self.norm(z)
        z = z + self.glu(z)
        return z

class SpatialMambaLike(nn.Module):
    """2D Mamba-like：Depthwise Separable Conv2d + Axial conv + GLU + 残差，输入 [B, L, p, p] → 汇聚到 [B, d]."""
    def __init__(self, L: int, d: int = 96, layers: int = 3, patch: int = 5):
        super().__init__()
        self.in_proj = nn.Conv2d(L, d, kernel_size=1)  # 频带压缩到 d 通道
        blks = []
        for _ in range(layers):
            blks += [
                # depthwise spatial
                nn.Conv2d(d, d, kernel_size=3, padding=1, groups=d),
                nn.Conv2d(d, d, kernel_size=1),
                nn.GELU(),
                # axial (H then W)
                nn.Conv2d(d, d, kernel_size=(1, 3), padding=(0, 1), groups=d),
                nn.Conv2d(d, d, kernel_size=1),
                nn.GELU(),
            ]
        self.blocks = nn.Sequential(*blks)
        self.out_proj = nn.Conv2d(d, d, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(d)
        self.glu = GLU(d)
    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(patch)  # [B,d,p,p]
        z = z + self.blocks(z)
        z = self.out_proj(z)
        z = self.pool(z).flatten(1)  # [B,d]
        z = self.norm(z)
        z = z + self.glu(z)
        return z

# ==== 5) 融合与解码 ====

class CrossFuse(nn.Module):
    """Early 门控 + Mid 互导（简化 Cross-Mamba）+ Late 融合头。输入/输出 [B, d]."""
    def __init__(self, d: int = 96):
        super().__init__()
        self.gs = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())
        self.gp = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())
        self.mid_s = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.mid_p = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.ln_s = nn.LayerNorm(d)
        self.ln_p = nn.LayerNorm(d)
        self.out = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, zs: torch.Tensor, zp: torch.Tensor) -> torch.Tensor:
        # Early gate
        zs = zs * self.gs(zp)
        zp = zp * self.gp(zs)
        # Mid 双向
        zs2 = self.ln_s(zs + self.mid_s(zp))
        zp2 = self.ln_p(zp + self.mid_p(zs))
        # Late
        zf = torch.cat([zs2, zp2], dim=-1)
        return self.out(zf)


class UnmixDecoder(nn.Module):
    """丰度头（Softmax 单纯形） + 端元扰动 (E = E0 + ΔE)；输出 A, E, Xhat。"""
    def __init__(self, d: int, K: int, L: int, E0: torch.Tensor | None = None):
        super().__init__()
        self.K, self.L = K, L
        self.abun = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, K)
        )
        if E0 is None:
            self.deltaE = nn.Parameter(torch.randn(K, L) * 1e-3)
            self.E0 = nn.Parameter(torch.zeros(K, L), requires_grad=False)
        else:
            with torch.no_grad():
                E0 = E0.clone().float()
                if E0.dim() == 2:
                    # 期望形状 [K, L]
                    if E0.size(0) != K and E0.size(1) == K:
                        E0 = E0.t()
                else:
                    raise ValueError("E0 必须为二维张量")
            self.E0 = nn.Parameter(E0, requires_grad=False)
            self.deltaE = nn.Parameter(torch.zeros_like(self.E0))

    def forward(self, zf: torch.Tensor):
        A = torch.softmax(self.abun(zf), dim=-1)  # [B,K]
        E = self.E0 + self.deltaE                 # [K,L]
        Xhat = A @ E                              # [B,L]
        return A, E, Xhat


# ==== 6) 总网络 ====
class DualBranchUnmixNet(nn.Module):
    def __init__(self, L: int, K: int, d: int = 96, ls: int = 3, lp: int = 3, patch: int = 5, E0: torch.Tensor | None = None, backend: str = 'mamba'):
        super().__init__()
        if backend == 'mamba' and HAS_MAMBA:
            self.spec = SpectralMambaReal(L=L, d=d, layers=ls)
            self.spa  = SpatialMambaReal(L=L, d=d, layers=lp, patch=patch)
        else:
            if backend != 'like' and not HAS_MAMBA:
                print("[Warning] mamba-ssm not found. Falling back to lightweight --backend like.")
            self.spec = SpectralMambaLike(L=L, d=d, layers=ls)
            self.spa  = SpatialMambaLike(L=L, d=d, layers=lp, patch=patch)
        self.fuse = CrossFuse(d=d)
        self.dec  = UnmixDecoder(d=d, K=K, L=L, E0=E0)

    def forward(self, patch: torch.Tensor, y_center: torch.Tensor | None = None):
        # patch: [B, L, p, p], y_center: [B, L] (可选，仅作为返回)
        B, L, p, _ = patch.shape
        ys = patch[:, :, p // 2, p // 2]  # 中心像素的光谱 [B, L]
        zs = self.spec(ys)                 # [B, d]
        zp = self.spa(patch)               # [B, d]
        zf = self.fuse(zs, zp)             # [B, d]
        A, E, Xhat = self.dec(zf)          # [B,K], [K,L], [B,L]
        return {"A": A, "E": E, "Xhat": Xhat, "Y": ys if y_center is None else y_center}


# ==== 7) 训练与评估 ====
@torch.no_grad()
def evaluate_epoch(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 50) -> Tuple[float, float]:
    model.eval()
    sad_list, rmse_list = [], []
    for b, (patch, y, _) in enumerate(loader):
        if b >= max_batches:
            break
        out = model(patch.to(device), y_center=y.to(device))
        Xhat = out["Xhat"]
        Y = out["Y"].to(device)
        sad = tensor_sad(Xhat, Y).mean().item()
        r = rmse(Xhat, Y)
        sad_list.append(sad)
        rmse_list.append(r)
    return float(sum(sad_list) / len(sad_list)), float(sum(rmse_list) / len(rmse_list))


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 载入数据（与你的 Data 保持一致）
    data_obj = Data(dataset=args.dataset, device='cpu')
    L = data_obj.get_L()
    K = data_obj.get_P()
    col = data_obj.get_col()  # H=W=col

    # 端元初始化（使用 M1 作为 E0）
    E0 = data_obj.get("init_weight")  # [L,K] 或 [K,L] 取决于 .mat
    if isinstance(E0, torch.Tensor):
        E0_t = E0.clone().detach().to(device).float()
        # 统一到 [K,L]
        if E0_t.dim() == 2:
            if E0_t.size(0) != K and E0_t.size(1) == K:
                E0_t = E0_t.t()
        else:
            raise ValueError("E0 维度错误")
    else:
        E0_t = None

    # 构建数据集/加载器
    train_set = PatchDataset(data_obj, patch=args.patch, stride=args.stride)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    # 模型
    model = DualBranchUnmixNet(L=L, K=K, d=args.embed_dim, ls=args.ls, lp=args.lp, patch=args.patch, E0=E0_t, backend=args.backend).to(device)

    # 优化器/调度器
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    # 损失权重
    lam_l1, lam_sad = args.lam_l1, args.lam_sad
    lam_sparse, lam_div, lam_e = args.lam_sparse, args.lam_div, args.lam_e

    best_sad = math.inf

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_l1 = run_sad = run_sparse = 0.0
        for step, (patch, y, _) in enumerate(train_loader, start=1):
            patch = patch.to(device)  # [B,L,p,p]
            y = y.to(device)          # [B,L]

            out = model(patch, y_center=y)
            Xhat, E, A = out["Xhat"], out["E"], out["A"]

            # 重构损失
            loss_l1 = F.l1_loss(Xhat, y)
            loss_sad = tensor_sad(Xhat, y).mean()

            # 稀疏与端元多样性
            loss_sparse = l05_sparsity(A)
            loss_div = endmember_diversity_loss(E)
            loss_e = (E - E.detach()).pow(2).mean() if E is out["E"] else torch.tensor(0.0, device=device)

            loss = lam_l1 * loss_l1 + lam_sad * loss_sad + lam_sparse * loss_sparse + lam_div * loss_div + lam_e * loss_e

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            run_l1 += loss_l1.item()
            run_sad += loss_sad.item()
            run_sparse += loss_sparse.item()

            if step % args.log_interval == 0:
                print(f"[Epoch {epoch}/{args.epochs}] Step {step}/{len(train_loader)} | L1={run_l1/args.log_interval:.4f} SAD={run_sad/args.log_interval:.4f} Sparse={run_sparse/args.log_interval:.4f}")
                run_l1 = run_sad = run_sparse = 0.0

        scheduler.step()

        # 简要评估（子集）
        val_sad, val_rmse = evaluate_epoch(model, train_loader, device, max_batches=args.eval_batches)
        print(f"=> Eval @Epoch {epoch}: SAD={val_sad:.4f}, RMSE={val_rmse:.4f}")

        # 保存最优
        if val_sad < best_sad:
            best_sad = val_sad
            os.makedirs(args.out_dir, exist_ok=True)
            ckpt_path = os.path.join(args.out_dir, f"best_{args.dataset}_dual_mamba.pth")
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optim_state': optim.state_dict(),
                'args': vars(args)
            }, ckpt_path)
            print(f"Saved best checkpoint to {ckpt_path}")

    print("Training finished. Best SAD:", best_sad)


def build_args():
    p = argparse.ArgumentParser()
    # Backend: 'mamba' requires mamba-ssm; 'like' uses built-in lightweight blocks
    p.add_argument('--backend', type=str, choices=['like','mamba'], default='mamba' if HAS_MAMBA else 'like')
    p.add_argument('--dataset', type=str, default='jasper')
    p.add_argument('--data_dir', type=str, default='./data')  # 与 Data 类保持一致的目录结构
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--seed', type=int, default=1234)

    # 训练
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)

    # Patch 提取
    p.add_argument('--patch', type=int, default=5)
    p.add_argument('--stride', type=int, default=1)

    # 模型结构
    p.add_argument('--embed_dim', type=int, default=96)
    p.add_argument('--ls', type=int, default=3, help='spectral layers')
    p.add_argument('--lp', type=int, default=3, help='spatial layers')

    # loss 权重
    p.add_argument('--lam_l1', type=float, default=1.0)
    p.add_argument('--lam_sad', type=float, default=0.5)
    p.add_argument('--lam_sparse', type=float, default=2e-4)
    p.add_argument('--lam_div', type=float, default=1e-2)
    p.add_argument('--lam_e', type=float, default=1e-3)

    # 评估与日志
    p.add_argument('--log_interval', type=int, default=50)
    p.add_argument('--eval_batches', type=int, default=50)
    p.add_argument('--out_dir', type=str, default='./checkpoints')
    return p.parse_args()


if __name__ == '__main__':
    args = build_args()

    # Data 类内部使用固定相对路径 ./data/{dataset}_dataset.mat
    # 因此只需确保文件存在即可；无需显式传 data_dir。若你的 Data 另有实现，请相应调整。
    train(args)
