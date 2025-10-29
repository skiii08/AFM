# ============================================================
# train_gat_dualphase_v2.py
# GAT-DualPhase (High-Accuracy, Extended Training + Embedding Trace)
# ============================================================

import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import HeteroData

from src.models.gat_dualphase import GATDualPhase


# ============================================================
# 0) Reproducibility
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 1) Paths
# ============================================================
PROJECT_ROOT = Path("/Users/watanabesaki/PycharmProjects/AFM")
DATA_PROCESSED = PROJECT_ROOT / "data/processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

TRAIN_PT = DATA_PROCESSED / "hetero_graph_train.pt"
VAL_PT = DATA_PROCESSED / "hetero_graph_val.pt"
MODEL_PATH = MODELS_DIR / "gat_dualphase_v2.pt"

USER_MEAN_IDX = 1
USER_STD_IDX = 2


# ============================================================
# 2) Hyperparams (Refined)
# ============================================================
HIDDEN_DIM = 128
HEADS = 2
DROPOUT = 0.30

LR = 1e-3
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 400
PATIENCE = 50

LOW_THRESH  = 4.0
HIGH_THRESH = 7.5
LOW_OVER_WEIGHT   = 2.2
HIGH_UNDER_WEIGHT = 0.7

LAMBDA_ASYM_INIT = 0.7
LAMBDA_ASYM_MIN  = 0.15

LAMBDA_CONTRAST = 0.6
MARGIN_COS = 0.8
CLIP_NORM = 5.0


# ============================================================
# 3) Utils
# ============================================================
def load_graph(path: Path) -> HeteroData:
    print(f"[Load] {path}")
    g = torch.load(path, weights_only=False)
    return g


@torch.no_grad()
def evaluate_val(model: GATDualPhase, graph: HeteroData, device: torch.device):
    model.eval()
    graph = graph.to(device)

    et = ("user", "review", "movie")
    y_z = graph[et].y.float()

    u_all = graph["user"].x
    mu_all = u_all[:, USER_MEAN_IDX]
    sd_all = u_all[:, USER_STD_IDX]
    sd_all = torch.where(sd_all == 0, torch.tensor(1.0, device=device), sd_all)

    y_pred_z, _ = model(graph)
    u_idx = graph[et].edge_index[0]
    mu = mu_all[u_idx]
    sd = sd_all[u_idx]

    y_true = torch.clamp(y_z * sd + mu, 1.0, 10.0)
    y_pred = torch.clamp(y_pred_z * sd + mu, 1.0, 10.0)

    rmse = torch.sqrt(F.mse_loss(y_pred, y_true)).item()
    mae  = F.l1_loss(y_pred, y_true).item()
    return rmse, mae


def cosine_contrast_loss(h_a: torch.Tensor, h_b: torch.Tensor, margin: float) -> tuple[torch.Tensor, float]:
    """
    満足(sat)と不満(dis)埋め込みをより明確に離すための強制版。
    - cos_sim > 1 - margin なら罰則を強化
    - cos_sim < 1 - margin でも軽く押し返す
    """
    h_a = F.normalize(h_a - h_a.mean(dim=0, keepdim=True), p=2, dim=-1)
    h_b = F.normalize(h_b - h_b.mean(dim=0, keepdim=True), p=2, dim=-1)
    cos_sim = (h_a * h_b).sum(dim=-1)

    # target: cos_sim を margin以下に抑える (例: margin=0.7 → cos_sim<0.3)
    target_max = 1.0 - margin
    loss_main = F.relu(cos_sim - target_max).mean()

    # small regularizer: 中心寄りにバランス
    loss_reg = (cos_sim.mean() ** 2)
    loss_total = loss_main + 0.05 * loss_reg

    return loss_total, cos_sim.mean().item()



# ============================================================
# 4) Main Train Loop
# ============================================================
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("GAT-DualPhase TRAINING v2 (asym + contrast + anneal + trace)")
    print("=" * 80)

    train_graph = load_graph(TRAIN_PT)
    val_graph   = load_graph(VAL_PT)

    user_dim  = train_graph["user"].x.size(1)
    movie_dim = train_graph["movie"].x.size(1)
    edge_dim  = train_graph["user", "review", "movie"].edge_attr.size(1)

    model = GATDualPhase(
        user_dim=user_dim,
        movie_dim=movie_dim,
        edge_dim=edge_dim,
        hidden_dim=HIDDEN_DIM,
        heads=HEADS,
        dropout=DROPOUT
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    et = ("user", "review", "movie")
    u_idx_tr = train_graph[et].edge_index[0]
    y_tr_z   = train_graph[et].y.float()
    u_all_tr = train_graph["user"].x
    mu_tr = u_all_tr[:, USER_MEAN_IDX][u_idx_tr]
    sd_tr = u_all_tr[:, USER_STD_IDX][u_idx_tr]
    sd_tr = torch.where(sd_tr == 0, torch.tensor(1.0), sd_tr)

    best_rmse = float("inf")
    patience_ctr = 0

    # ========================================================
    # Training Loop
    # ========================================================
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        prog = (epoch - 1) / max(1, NUM_EPOCHS - 1)
        lambda_asym = max(LAMBDA_ASYM_MIN, LAMBDA_ASYM_INIT * (1.0 - prog))

        train_graph = train_graph.to(device)
        y_pred_z, (u_sat, u_dis, m_sat, m_dis) = model(train_graph)

        # Core
        y_tr_z_dev = y_tr_z.to(device)
        loss_core = F.mse_loss(y_pred_z, y_tr_z_dev)

        # RAW-scale
        mu_tr_dev = mu_tr.to(device)
        sd_tr_dev = sd_tr.to(device)
        y_true_raw = torch.clamp(y_tr_z_dev * sd_tr_dev + mu_tr_dev, 1.0, 10.0)
        y_pred_raw = torch.clamp(y_pred_z * sd_tr_dev + mu_tr_dev, 1.0, 10.0)

        low_mask  = (y_true_raw <= LOW_THRESH)
        high_mask = (y_true_raw >= HIGH_THRESH)
        over_low  = torch.relu(y_pred_raw - y_true_raw)
        under_high = torch.relu(y_true_raw - y_pred_raw)

        loss_low  = (over_low[low_mask] ** 2).mean() if low_mask.any() else torch.tensor(0.0, device=device)
        loss_high = (under_high[high_mask] ** 2).mean() if high_mask.any() else torch.tensor(0.0, device=device)
        loss_asym = lambda_asym * (LOW_OVER_WEIGHT * loss_low + HIGH_UNDER_WEIGHT * loss_high)

        # Contrast + trace
        loss_ctr_u, cos_u = cosine_contrast_loss(u_sat, u_dis, margin=MARGIN_COS)
        loss_ctr_m, cos_m = cosine_contrast_loss(m_sat, m_dis, margin=MARGIN_COS)
        loss_ctr = LAMBDA_CONTRAST * (loss_ctr_u + loss_ctr_m)

        loss_total = loss_core + loss_asym + loss_ctr
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()

        rmse, mae = evaluate_val(model, val_graph, device)

        print(
            f"Epoch {epoch:03d} | "
            f"Loss={loss_total.item():.4f} (core={loss_core.item():.4f}, asym={loss_asym.item():.4f}, ctr={loss_ctr.item():.4f}) "
            f"| Val RMSE={rmse:.4f} MAE={mae:.4f} "
            f"| λ_asym={lambda_asym:.3f} | cos_u={cos_u:.3f}, cos_m={cos_m:.3f}"
        )

        # Early Stop
        if rmse < best_rmse - 1e-5:
            best_rmse = rmse
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✅ Saved best model: RMSE={best_rmse:.4f}")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n[EarlyStop] RMSE stagnant. Best RMSE={best_rmse:.4f}")
                break


if __name__ == "__main__":
    main()
