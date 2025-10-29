# ============================================================
# EVALUATE SimpleGAT (10-point scale, AFM-compatible)
# ============================================================

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from src.models.gat_simple import SimpleGAT


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
PROJECT_ROOT = Path("/Users/watanabesaki/PycharmProjects/AFM")
DATA_PROCESSED = PROJECT_ROOT / "data/processed"
TEST_PT = DATA_PROCESSED / "hetero_graph_test.pt"
MODEL_PATH = PROJECT_ROOT / "models" / "gat_simple.pt"


# ------------------------------------------------------------
# Evaluate (Raw scale)
# ------------------------------------------------------------
@torch.no_grad()
def evaluate_gat(model, data):
    model.eval()
    out = model(data)
    et = ('user', 'review', 'movie')
    y_z = data[et].y.float()

    mu = data['user'].x[data[et].edge_index[0], 1]
    sd = data['user'].x[data[et].edge_index[0], 2]
    sd = torch.where(sd == 0, torch.tensor(1.0, device=sd.device), sd)

    y_true = torch.clamp(y_z * sd + mu, 1.0, 10.0)
    y_pred = torch.clamp(out * sd + mu, 1.0, 10.0)

    rmse = torch.sqrt(F.mse_loss(y_pred, y_true)).item()
    mae = F.l1_loss(y_pred, y_true).item()

    # 帯域別詳細
    df = pd.DataFrame({
        "y_true": y_true.cpu().numpy(),
        "y_pred": y_pred.cpu().numpy()
    })
    bands = {
        "Low (1.0-4.0)": (1.0, 4.0),
        "Mid (4.1-7.0)": (4.1, 7.0),
        "High (7.1-10.0)": (7.1, 10.0),
    }
    detail = {}
    for k, (lo, hi) in bands.items():
        sub = df[(df.y_true >= lo) & (df.y_true <= hi)]
        if len(sub) == 0:
            detail[k] = {"Count": 0, "RMSE": np.nan, "MAE": np.nan}
        else:
            r = float(np.sqrt(np.mean((sub.y_pred - sub.y_true) ** 2)))
            m = float(np.mean(np.abs(sub.y_pred - sub.y_true)))
            detail[k] = {"Count": int(len(sub)), "RMSE": r, "MAE": m}

    return rmse, mae, detail


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print("=" * 80)
    print("SimpleGAT EVALUATION (10-point scale)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(TEST_PT, weights_only=False).to(device)

    # 実データから自動で次元取得
    user_dim = data['user'].x.size(1)
    movie_dim = data['movie'].x.size(1)
    edge_dim = data['user', 'review', 'movie'].edge_attr.size(1)
    print(f"[DEBUG] user_dim={user_dim}, movie_dim={movie_dim}, edge_dim={edge_dim}")

    # モデル構築
    model = SimpleGAT(user_dim=user_dim, movie_dim=movie_dim, edge_dim=edge_dim).to(device)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("✅ モデル読み込み成功。")
    except Exception as e:
        print(f"❌ モデル読み込み失敗: {e}")
        return

    rmse, mae, detail = evaluate_gat(model, data)

    # --------------------------------------------------------
    # 結果出力
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("【総合評価結果 (Test Set - 10点満点)】")
    print("-" * 60)
    total_count = sum(d["Count"] for d in detail.values())
    print(f"📝 総エッジ数: {total_count}")
    print(f"🚀 Total RMSE: {rmse:.4f}")
    print(f"🎯 Total MAE:  {mae:.4f}")
    print("=" * 60)

    print("\n【Rating帯域別 詳細分析 (10点満点)】")
    print("-" * 60)
    df = pd.DataFrame.from_dict(detail, orient="index")
    df.index.name = "Rating Band"
    print(df.to_string(float_format="%.4f"))
    print("-" * 60)


if __name__ == "__main__":
    main()
