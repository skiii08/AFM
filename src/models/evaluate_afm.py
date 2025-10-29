# src/models/evaluate_afm.py
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from torch_geometric.data import HeteroData

# ← v2 の AFM をそのまま使う（ローカル再定義はしない）
from src.models.afm import AFM

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
TEST_GRAPH_PT  = DATA_PROCESSED / "hetero_graph_test.pt"
MODEL_PATH     = PROJECT_ROOT / "models" / "afm_model.pt"

# 仕様準拠（ユーザー特徴内のμ・σの位置）
USER_MEAN_ABS_INDEX   = 1
USER_STD_ABS_INDEX    = 2
EDGE_LEAK_START_INDEX = 36

# 評価時のモデルハイパラ（学習と一致させる）
K_EMBED   = 32
ATTN_SIZE = 128
DROPOUT   = 0.2


@torch.no_grad()
def evaluate_model(model, graph: HeteroData):
    model.eval()
    et = ('user', 'review', 'movie')
    y_z = graph[et].y.float()
    u_idx = graph[et].edge_index[0]
    m_idx = graph[et].edge_index[1]

    u_all = graph['user'].x
    m_all = graph['movie'].x
    e_full = graph[et].edge_attr

    # μ/σ 抜き出し → スライスで除去（219D化）
    mu_all = u_all[:, USER_MEAN_ABS_INDEX]
    sd_all = u_all[:, USER_STD_ABS_INDEX]
    sd_all = torch.where(sd_all == 0, torch.tensor(1.0, dtype=torch.float32), sd_all)

    before = u_all[:, :USER_MEAN_ABS_INDEX]
    after  = u_all[:, USER_STD_ABS_INDEX + 1:]
    u_sliced = torch.cat([before, after], dim=1)  # (N_u, 219)

    # edge毎にインデックス
    u_feat = u_sliced[u_idx]
    m_feat = m_all[m_idx]
    e_attr = e_full[:, :EDGE_LEAK_START_INDEX] if e_full.size(1) >= EDGE_LEAK_START_INDEX else e_full

    # 予測（Z）→ 逆正規化
    z_hat = model(u_feat, m_feat, e_attr)
    mu = mu_all[u_idx]
    sd = sd_all[u_idx]
    y_true = torch.clamp(y_z * sd + mu, 1.0, 10.0)
    y_pred = torch.clamp(z_hat * sd + mu, 1.0, 10.0)

    rmse = torch.sqrt(F.mse_loss(y_pred, y_true)).item()
    mae  = F.l1_loss(y_pred, y_true).item()

    # 帯域別
    df = pd.DataFrame({"y_true": y_true.cpu().numpy(), "y_pred": y_pred.cpu().numpy()})
    bands = {
        "Low (1.0-4.0)"   : (1.0, 4.0),
        "Mid (4.1-7.0)"   : (4.1, 7.0),
        "High (7.1-10.0)" : (7.1, 10.0),
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


def main():
    print("="*80)
    print("07_EVALUATE_AFM (Sentiment-Conditioned Attention, v2 architecture)")
    print("="*80)

    if not MODEL_PATH.exists() or not TEST_GRAPH_PT.exists():
        print("❌ エラー: 必要ファイルが見つかりません。")
        return

    graph = torch.load(TEST_GRAPH_PT, weights_only=False)

    # 次元は実データから推定
    u_all = graph['user'].x
    m_all = graph['movie'].x
    e_all = graph['user','review','movie'].edge_attr
    Du = (u_all.size(1) - 2)  # μ,σを抜くので -2 → 219
    Dm = m_all.size(1)        # 636
    De = min(e_all.size(1), EDGE_LEAK_START_INDEX)  # 36

    # v2 のコンストラクタ引数名に合わせる
    model = AFM(
        user_dim=Du,
        movie_dim=Dm,
        edge_dim=De,
        embedding_dim=K_EMBED,
        attn_size=ATTN_SIZE,
        dropout=DROPOUT,
    )

    # 確認用：v2 なら attn 入力は 66 (= 32 + 16 + 18)
    try:
        _ = model.state_dict()  # just touch
        print(f"[Check] AFM v2 loaded (expect attn_input_dim=66)")
    except Exception as e:
        print(f"❌ モデル構築失敗: {e}")
        return

    try:
        state = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state)
        print("✅ モデル重みの読み込みに成功しました（afm.py v2 と一致）。")
    except Exception as e:
        print(f"❌ モデル読み込み失敗: {e}")
        return

    rmse, mae, detail = evaluate_model(model, graph)

    print("\n" + "="*60)
    print("【総合評価結果 (Test Set - 10点満点)】")
    print("-"*60)
    total_count = sum(d["Count"] for d in detail.values())
    print(f"📝 総エッジ数: {total_count}")
    print(f"🚀 Total RMSE: {rmse:.4f}")
    print(f"🎯 Total MAE:  {mae:.4f}")
    print("="*60)

    print("\n【Rating帯域別 詳細分析 (10点満点)】")
    print("-"*60)
    df = pd.DataFrame.from_dict(detail, orient="index")
    df.index.name = "Rating Band"
    print(df.to_string(float_format="%.4f"))
    print("-" * 60)


if __name__ == "__main__":
    main()
