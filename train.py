"""
Dual-branch (Spectral/Spatial) Mamba-like Autoencoder for Hyperspectral Unmixing
Dataset: Jasper Ridge (L=198, K=4, H=W=100 assumed by provided loader)

- 数据载入：沿用你的 Data 类（datasets.py）
- PatchDataset：基于 Data.get('hs_img') 重建 H×W×L 立方体，再按滑窗提供 patch
- 模型：光谱分支(1D Mamba-like) + 空间分支(2D Mamba-like) + 三阶段融合 + 可解释解码头(A@E)
- 损失：L1 + SAD + 稀疏(ℓ0.5) + 端元多样性；（TV 可选，默认关闭）
- 说明：本实现支持 `--backend mamba`（需 `pip install mamba-ssm`）与 `--backend like`（内置轻量近似）。若你已装 VSSM 也可按同形状替换。

运行示例：
    python train.py \
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
import numpy as np
import os
from plots import plot_abundance, plot_endmembers

# tqdm progress bar (optional)
try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False
    def tqdm(iterable=None, **kwargs):
        # fallback: return the iterable unchanged for compatibility
        return iterable if iterable is not None else []

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


def logdet_volume_loss(E: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Encourage endmember simplex spread without GT by maximizing volume.
    We minimize -logdet(E E^T + eps I). E: [K,L].
    """
    K = E.size(0)
    G = E @ E.t()
    G = G + eps * torch.eye(K, dtype=E.dtype, device=E.device)
    return -torch.logdet(G)


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
        zs1 = zs * self.gs(zp)
        zp1 = zp * self.gp(zs)
        # Mid 双向
        zs2 = self.ln_s(zs1 + self.mid_s(zp1))
        zp2 = self.ln_p(zp1 + self.mid_p(zs1))
        # Late
        zf = torch.cat([zs2, zp2], dim=-1)
        return self.out(zf)


class UnmixDecoder(nn.Module):
    """Abundance head (simplex via softmax) + endmember perturbation (E=E0+ΔE).
    No GT is used; apply non-negativity to E and temperature on A.
    """
    def __init__(self, d: int, K: int, L: int, E0: torch.Tensor | None = None, tau: float = 1.0, nonneg_E: bool = True):
        super().__init__()
        self.K, self.L = K, L
        self.tau = tau
        self.nonneg_E = nonneg_E
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
                    if E0.size(0) != K and E0.size(1) == K:
                        E0 = E0.t()
                else:
                    raise ValueError("E0 必须为二维张量")
            self.E0 = nn.Parameter(E0, requires_grad=False)
            self.deltaE = nn.Parameter(torch.zeros_like(self.E0))

    def forward(self, zf: torch.Tensor):
        A = torch.softmax(self.abun(zf) / self.tau, dim=-1)  # [B,K]
        E = self.E0 + self.deltaE                             # [K,L]
        if self.nonneg_E:
            E = F.softplus(E)                                 # 非负约束
        Xhat = A @ E                                          # [B,L]
        return A, E, Xhat


# ==== 6) 总网络 ====
class DualBranchUnmixNet(nn.Module):
    def __init__(self, L: int, K: int, d: int = 96, ls: int = 3, lp: int = 3, patch: int = 5, E0: torch.Tensor | None = None, backend: str = 'mamba', tau: float = 1.0):
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
        self.dec  = UnmixDecoder(d=d, K=K, L=L, E0=E0, tau=tau, nonneg_E=True)

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
    for b, batch in enumerate(loader):
        if b >= max_batches:
            break
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            patch, y, _ = batch
        else:
            patch, y = batch
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

    # 端元初始化（使用 M1 作为 E0；来自 Y 的预解混，不是 GT）
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
    model = DualBranchUnmixNet(L=L, K=K, d=args.embed_dim, ls=args.ls, lp=args.lp, patch=args.patch, E0=E0_t, backend=args.backend, tau=args.tau).to(device)

    # 优化器/调度器
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    # 损失权重 (using args directly now)
    best_sad = math.inf

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_l1 = run_sad = run_sparse = 0.0
        # create an iterator and optionally wrap with tqdm for a progress bar
        iterator = enumerate(train_loader, start=1)
        if HAS_TQDM:
            pbar = tqdm(iterator, total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs}", unit="step")
        else:
            pbar = iterator
        for step, batch in pbar:
            # 支持 (patch, y) 或 (patch, y, a_true)，训练阶段忽略 a_true
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                patch, y, _ = batch
            else:
                patch, y = batch
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
            loss_vol = logdet_volume_loss(E)
            # 可选：若允许用 M1 作为来自 Y 的先验，可启用弱锚定（默认关闭）
            loss_esam = 0.0
            if args.lam_esam > 0.0 and hasattr(model.dec, "E0") and model.dec.E0 is not None:
                Eh = F.normalize(E, dim=-1); Er = F.normalize(model.dec.E0, dim=-1)
                cos = (Eh * Er).sum(dim=-1).clamp(-1.0, 1.0)
                loss_esam = torch.acos(cos).mean()

            loss = (
                args.lam_l1 * loss_l1 +
                args.lam_sad * loss_sad +
                args.lam_sparse * loss_sparse +
                args.lam_div * loss_div +
                args.lam_vol * loss_vol +
                args.lam_esam * loss_esam
            )
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            run_l1 += loss_l1.item()
            run_sad += loss_sad.item()
            run_sparse += loss_sparse.item()

            if step % args.log_interval == 0:
                avg_l1 = run_l1 / args.log_interval
                avg_sad = run_sad / args.log_interval
                avg_sparse = run_sparse / args.log_interval
                if HAS_TQDM:
                    # show averaged metrics on the progress bar
                    pbar.set_postfix({"L1": f"{avg_l1:.4f}", "SAD": f"{avg_sad:.4f}", "Sparse": f"{avg_sparse:.4f}"})
                else:
                    # fallback: print as before
                    print(f"[Epoch {epoch}/{args.epochs}] Step {step}/{len(train_loader)} | L1={avg_l1:.4f} SAD={avg_sad:.4f} Sparse={avg_sparse:.4f}")
                run_l1 = run_sad = run_sparse = 0.0
        if HAS_TQDM:
            try:
                pbar.close()
            except Exception:
                pass

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
    # ---- 在训练结束后，对整图进行推断并绘图 ----
    try:
        os.makedirs(os.path.join(args.out_dir, 'plots'), exist_ok=True)
        ckpt_path = os.path.join(args.out_dir, f"best_{args.dataset}_dual_mamba.pth")
        if os.path.exists(ckpt_path):
            print(f"Loading best checkpoint from {ckpt_path} for full-image inference...")
            ck = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(ck['model_state'])
        else:
            print("No checkpoint found, using current model weights for inference.")

        # 使用 Data 的原始整图（Y: [N,L], A: [N,K]）来生成完整丰度图
        data_obj = Data(dataset=args.dataset, device='cpu')
        Y = data_obj.get('hs_img')  # [N,L]
        A_true = data_obj.get('abd_map')  # [N,K]
        L = data_obj.get_L()
        K = data_obj.get_P()
        col = data_obj.get_col()
        H = W = col

        model.to('cpu')
        model.eval()

        # 将整图分批推断（按像素中心patch），恢复为 HxWxK 的丰度图
        patch = args.patch
        r = patch // 2
        # 重建立方体 [H,W,L]
        cube = Y.view(H, W, L)
        abd_spa_map = np.zeros((H, W, K), dtype=np.float32)
        abd_spr_map = np.zeros((H, W, K), dtype=np.float32)
        abd_fused_map = np.zeros((H, W, K), dtype=np.float32)

        # 对每个像素提取 patch 并推断（会比较慢；可优化为批量）
        pts = []
        coords = []
        for i in range(r, H - r):
            for j in range(r, W - r):
                p = cube[i - r:i + r + 1, j - r:j + r + 1, :].permute(2, 0, 1).unsqueeze(0)  # [1,L,p,p]
                pts.append(p)
                coords.append((i, j))

        # 分批运行以节省显存
        batch = 256
        for s in range(0, len(pts), batch):
            batch_pts = torch.cat(pts[s:s+batch], dim=0)  # [B,L,p,p]
            with torch.no_grad():
                out = model(batch_pts)
            A_pred = out['A'].cpu().numpy()  # [B,K]
            # 目前 model 的内部没有拆分 spa/spr 分支输出；我们只能使用 fused A
            for idx, (ii, jj) in enumerate(coords[s:s+batch]):
                abd_fused_map[ii, jj, :] = A_pred[idx]

        # 保存真实和估计的丰度图（整图形式）以及端元估计
        plots_dir = os.path.join(args.out_dir, 'plots')
        np.save(os.path.join(plots_dir, 'abd_true.npy'), A_true.view(H, W, K).numpy())
        np.save(os.path.join(plots_dir, 'abd_fused.npy'), abd_fused_map)

        # 获取端元（E）并保存：解码器 E 是 model.dec.E0 + deltaE
        E = model.dec.E0.detach().cpu().numpy() + model.dec.deltaE.detach().cpu().numpy()
        # spa/spr 端元暂不可用（模型未暴露），用相同 E 填充占位
        plot_abundance(A_true.view(H, W, K).numpy(), abd_fused_map, abd_fused_map, abd_fused_map, K, plots_dir)
        # 为端元绘图准备输入：target M: [L,K] 或 [K,L]
        M = data_obj.get('end_mem')
        # 规范 M 维度到 [L,K]
        M_np = M.numpy()
        if M_np.shape[0] == K and M_np.shape[1] == L:
            M_np = M_np.T
        plot_endmembers(M_np, E.T if E.shape[0] == K and E.shape[1] == L else E, E, E, K, plots_dir)
        print(f"Saved full-image plots to {plots_dir}")

    # ---- 计算“真实”的指标：端元 SAD 与 丰度/重构 RMSE、重构 SAD ----
        try:
            import itertools
            # 1) 端元 SAD（与真端元做最佳匹配，置换不变）
            #   规范维度：E_hat: [K,L], M_true: [K,L]
            E_hat = model.dec.E0.detach().cpu().numpy() + model.dec.deltaE.detach().cpu().numpy()  # [K,L]
            M_true_np = data_obj.get('end_mem').numpy()  # [L,K] 或 [K,L]
            if M_true_np.shape == (L, K):
                M_true = M_true_np.T
            elif M_true_np.shape == (K, L):
                M_true = M_true_np
            else:
                raise ValueError(f"Unexpected end_mem shape: {M_true_np.shape}, expected (L,K) or (K,L)")

            def sad_vec(u: np.ndarray, v: np.ndarray, eps: float = 1e-8) -> float:
                u = u.astype(np.float64)
                v = v.astype(np.float64)
                un = u / (np.linalg.norm(u) + eps)
                vn = v / (np.linalg.norm(v) + eps)
                cos = np.clip((un * vn).sum(), -1.0, 1.0)
                return float(np.arccos(cos))

            # 计算 KxK 的 SAD 代价矩阵
            cost = np.zeros((K, K), dtype=np.float64)
            for i in range(K):
                for j in range(K):
                    cost[i, j] = sad_vec(E_hat[i], M_true[j])

            # 对 K 较小（如 4-6），用遍历置换找最小总代价
            best_perm, best_sum = None, float('inf')
            for perm in itertools.permutations(range(K)):
                s = sum(cost[i, perm[i]] for i in range(K))
                if s < best_sum:
                    best_sum = s
                    best_perm = perm
            sad_per_em = [cost[i, best_perm[i]] for i in range(K)]
            sad_mean_rad = float(np.mean(sad_per_em))
            sad_mean_deg = float(np.degrees(sad_mean_rad))

            # 2) 丰度 RMSE（仅在有效中心区域上计算，以避免边缘未推断像素影响）
            A_true_full = A_true.view(H, W, K).numpy()
            mask = np.zeros((H, W), dtype=bool)
            mask[r:H - r, r:W - r] = True
            A_true_valid = A_true_full[mask]        # [N_valid, K]
            A_pred_valid = abd_fused_map[mask]      # [N_valid, K]
            abd_rmse = float(np.sqrt(np.mean((A_true_valid - A_pred_valid) ** 2)))

            # 3) 重构谱 RMSE/SAD（真实整图中心像素区域）
            cube_np = cube.numpy()                  # [H,W,L]
            Y_true_valid = cube_np[mask]           # [N_valid, L]
            Xhat_pred_valid = A_pred_valid @ E_hat # [N_valid, L]
            recon_rmse = float(np.sqrt(np.mean((Y_true_valid - Xhat_pred_valid) ** 2)))
            recon_sad_list = [sad_vec(Xhat_pred_valid[i], Y_true_valid[i]) for i in range(Y_true_valid.shape[0])]
            recon_sad_mean_rad = float(np.mean(recon_sad_list))
            recon_sad_mean_deg = float(np.degrees(recon_sad_mean_rad))

            # 打印并保存
            print(
                f"=> True metrics: Recon RMSE={recon_rmse:.6f}, Recon SAD={recon_sad_mean_rad:.6f} rad ({recon_sad_mean_deg:.2f} deg); "
                f"Abundance RMSE={abd_rmse:.6f}, Endmember SAD={sad_mean_rad:.6f} rad ({sad_mean_deg:.2f} deg)"
            )
            with open(os.path.join(plots_dir, 'metrics.txt'), 'w') as f:
                f.write(f"Reconstruction RMSE: {recon_rmse:.6f}\n")
                f.write(f"Reconstruction SAD (mean): {recon_sad_mean_rad:.6f} rad ({recon_sad_mean_deg:.2f} deg)\n")
                f.write(f"Abundance RMSE: {abd_rmse:.6f}\n")
                f.write(f"Endmember SAD (mean): {sad_mean_rad:.6f} rad ({sad_mean_deg:.2f} deg)\n")
                f.write("Endmember SAD per pair (rad) in matched order: " + ", ".join(f"{x:.6f}" for x in sad_per_em) + "\n")
        except Exception as me:
            print("Metrics computation failed:", me)
    except Exception as e:
        print("Full-image plotting failed:", e)


def build_args():
    p = argparse.ArgumentParser()
    # Backend: 'mamba' requires mamba-ssm; 'like' uses built-in lightweight blocks
    p.add_argument('--backend', type=str, choices=['like','mamba'], default='mamba' if HAS_MAMBA else 'like')
    p.add_argument('--dataset', type=str, choices=['samson','jasper','urban','apex','dc','moffett'], default='jasper')
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
    p.add_argument('--lam_div', type=float, default=1e-3)
    p.add_argument('--lam_vol', type=float, default=5e-4)
    p.add_argument('--tau', type=float, default=0.8)
    p.add_argument('--lam_esam', type=float, default=0.0)  # optional weak anchor to M1 (off by default)

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
