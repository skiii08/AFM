# src/models/train_afm.py
# ============================================================
# AFM MODEL TRAINING (Sentiment-Conditioned Attention + Asymmetric Penalty)
# ============================================================

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# モデルを外部定義から読み込む（← これが重要！）
from src.models.afm import AFM


# ==============================
# 0. 再現性
# ==============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ==============================
# 1. パスとハイパーパラメータ
# ==============================
PROJECT_ROOT = Path("/Users/watanabesaki/PycharmProjects/AFM")
DATA_PROCESSED = PROJECT_ROOT / "data/processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

TRAIN_PT = DATA_PROCESSED / "hetero_graph_train.pt"
VAL_PT = DATA_PROCESSED / "hetero_graph_val.pt"
MODEL_PATH = MODELS_DIR / "afm_model.pt"

# ---- モデル構造 ----
USER_DIM = 219
MOVIE_DIM = 636
EDGE_DIM = 36
K_EMBED = 32
ATTN_SIZE = 128
DROPOUT = 0.2

# ---- 学習ハイパラ ----
NUM_EPOCHS = 250
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-5

# ---- Hard Negative Sampling（低評価の出現率を上げる）----
USE_WEIGHTED_SAMPLER = True
LOW_RATING_THRESHOLD = 4.0
LOW_SAMPLE_MULTIPLIER = 3.0

# ---- 非対称ペナルティ ----
USE_ASYM_LOSS = True
LAMBDA_ASYM_INIT = 0.6
LAMBDA_ASYM_MIN = 0.05
HIGH_RATING_THRESHOLD = 7.5
HIGH_UNDER_PENALTY_RATIO = 0.5

# ---- 早期終了 ----
PATIENCE = 22


# ==============================
# 2. データロード関数
# ==============================
USER_MEAN_IDX = 1
USER_STD_IDX = 2
EDGE_LEAK_START = 36


def load_graph(path: Path):
    data = torch.load(path, weights_only=False)

    u_all = data["user"].x
    m_all = data["movie"].x
    e = data["user", "review", "movie"]
    u_idx, m_idx = e.edge_index
    e_attr_full = e.edge_attr
    y_norm = e.y.float()

    # μ・σ抽出
    u_mu = u_all[:, USER_MEAN_IDX]
    u_sigma = u_all[:, USER_STD_IDX]
    u_sigma = torch.where(u_sigma == 0, torch.tensor(1.0, dtype=torch.float32), u_sigma)

    # μ・σ除去（219次元化）
    before_mu = u_all[:, :USER_MEAN_IDX]
    after_sigma = u_all[:, USER_STD_IDX + 1 :]
    u_sliced = torch.cat([before_mu, after_sigma], dim=1)

    # Edge単位に抽出
    u_feat = u_sliced[u_idx]
    m_feat = m_all[m_idx]
    e_attr = e_attr_full[:, :EDGE_LEAK_START]

    mu = u_mu[u_idx]
    sigma = u_sigma[u_idx]
    y_raw = y_norm * sigma + mu

    return u_feat, m_feat, e_attr, y_norm, y_raw, mu, sigma


# ==============================
# 3. 検証関数（Rawスケール）
# ==============================
@torch.no_grad()
def evaluate(model, u, m, e_attr, y_norm, mu, sigma):
    model.eval()
    y_pred_norm = model(u, m, e_attr)
    y_pred_raw = torch.clamp(y_pred_norm * sigma + mu, 1.0, 10.0)
    y_true_raw = torch.clamp(y_norm * sigma + mu, 1.0, 10.0)
    rmse = torch.sqrt(F.mse_loss(y_pred_raw, y_true_raw)).item()
    mae = F.l1_loss(y_pred_raw, y_true_raw).item()
    return rmse, mae


# ==============================
# 4. メイン学習ループ
# ==============================
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("AFM MODEL TRAINING (Sentiment-Conditioned Attention + Asymmetric Penalty)")
    print("=" * 80)

    # ---- Load ----
    print("[Load] TRAIN")
    u_tr, m_tr, e_tr, y_tr_norm, y_tr_raw, mu_tr, sigma_tr = load_graph(TRAIN_PT)
    print(f"  [DEBUG] u_feat shape: {u_tr.shape} | m_feat: {m_tr.shape} | e_attr: {e_tr.shape}")

    print("[Load] VAL  ")
    u_va, m_va, e_va, y_va_norm, y_va_raw, mu_va, sigma_va = load_graph(VAL_PT)
    print(f"  [DEBUG] u_feat shape: {u_va.shape} | m_feat: {m_va.shape} | e_attr: {e_va.shape}")

    # ---- Dataset / Sampler ----
    train_ds = TensorDataset(u_tr, m_tr, e_tr, y_tr_norm, y_tr_raw, mu_tr, sigma_tr)
    if USE_WEIGHTED_SAMPLER:
        weights = torch.ones(len(y_tr_raw), dtype=torch.float32)
        weights[y_tr_raw <= LOW_RATING_THRESHOLD] = LOW_SAMPLE_MULTIPLIER
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, drop_last=False)
    else:
        train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    # ---- モデル定義 ----
    model = AFM(
        user_dim=USER_DIM,
        movie_dim=MOVIE_DIM,
        edge_dim=EDGE_DIM,
        embedding_dim=K_EMBED,
        attn_size=ATTN_SIZE,
        dropout=DROPOUT,
    ).to(device)

    print(f"[Init] AFM(user={USER_DIM}, movie={MOVIE_DIM}, edge={EDGE_DIM})  k={K_EMBED}, attn={ATTN_SIZE}")

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_rmse = float("inf")
    patience_ctr = 0

    # ---- Training ----
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_samples = 0

        # λ_asym を徐々に減衰
        if USE_ASYM_LOSS:
            prog = (epoch - 1) / max(1, NUM_EPOCHS - 1)
            lambda_asym = max(LAMBDA_ASYM_MIN, LAMBDA_ASYM_INIT * (1.0 - prog))
        else:
            lambda_asym = 0.0

        for u_b, m_b, e_b, y_norm_b, y_raw_b, mu_b, sigma_b in train_ld:
            u_b, m_b, e_b = u_b.to(device), m_b.to(device), e_b.to(device)
            y_norm_b, y_raw_b = y_norm_b.to(device), y_raw_b.to(device)
            mu_b, sigma_b = mu_b.to(device), sigma_b.to(device)

            optimizer.zero_grad()

            # 予測（Zスコア）
            y_pred_norm_b = model(u_b, m_b, e_b)

            # コア損失（ZスコアMSE）
            loss_core = F.mse_loss(y_pred_norm_b, y_norm_b)

            # Rawスケールで非対称ペナルティ
            y_pred_raw_b = torch.clamp(y_pred_norm_b * sigma_b + mu_b, 1.0, 10.0)
            if USE_ASYM_LOSS:
                low_mask = y_raw_b <= LOW_RATING_THRESHOLD
                over_low = torch.relu(y_pred_raw_b - y_raw_b)
                loss_asym_low = (over_low[low_mask] ** 2).mean() if low_mask.any() else torch.tensor(0.0, device=device)

                high_mask = y_raw_b >= HIGH_RATING_THRESHOLD
                under_high = torch.relu(y_raw_b - y_pred_raw_b)
                loss_asym_high = (under_high[high_mask] ** 2).mean() if high_mask.any() else torch.tensor(0.0, device=device)

                loss_asym = lambda_asym * (loss_asym_low + HIGH_UNDER_PENALTY_RATIO * loss_asym_high)
            else:
                loss_asym = torch.tensor(0.0, device=device)

            loss = loss_core + loss_asym
            loss.backward()
            optimizer.step()

            bs = u_b.size(0)
            epoch_loss += loss.item() * bs
            n_samples += bs

        epoch_loss /= max(1, n_samples)

        # ---- Validation ----
        rmse, mae = evaluate(
            model,
            u_va.to(device),
            m_va.to(device),
            e_va.to(device),
            y_va_norm.to(device),
            mu_va.to(device),
            sigma_va.to(device),
        )

        print(
            f"Epoch {epoch:03d} | TrainLoss: {epoch_loss:.5f} | Val RMSE: {rmse:.4f} | "
            f"Val MAE: {mae:.4f} | λ_asym: {lambda_asym:.3f}"
        )

        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  -> Saved: {MODEL_PATH.name} (best RMSE {best_rmse:.4f})")
            patience_ctr = 0
        else:
            patience_ctr += 1
            print(f"  -> Patience {patience_ctr}/{PATIENCE}")
            if patience_ctr >= PATIENCE:
                print("\n[EarlyStop] No improvement. Stop.")
                break


if __name__ == "__main__":
    main()
