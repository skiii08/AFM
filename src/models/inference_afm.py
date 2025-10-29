# src/models/inference_afm.py (最終確定版: 198Dモデルとハイパーパラメータに合わせて修正)

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import pandas as pd
import random
import os
import torch.nn.functional as F


# ===================================================================
# 1. AFM MODEL DEFINITION (Z-Score出力モデル - Clampを削除)
# ===================================================================

class AFM(nn.Module):
    def __init__(self, user_dim, movie_dim, edge_dim,
                 embedding_dim=32, # 💥 修正: K_EMBED=32
                 attn_size=32,     # 💥 修正: ATTN_SIZE=32
                 dropout_rate=0.2):
        super().__init__()

        self.user_embed = nn.Linear(user_dim, embedding_dim, bias=False)
        self.movie_embed = nn.Linear(movie_dim, embedding_dim, bias=False)

        attn_input_dim = embedding_dim + edge_dim
        self.attn_layer = nn.Sequential(
            nn.Linear(attn_input_dim, attn_size),
            nn.ReLU(),
            nn.Linear(attn_size, 1, bias=False)
        )
        self.dropout = nn.Dropout(dropout_rate)

        self.bias = nn.Parameter(torch.zeros(1))
        self.linear_u = nn.Linear(user_dim, 1, bias=False)
        self.linear_m = nn.Linear(movie_dim, 1, bias=False)

    def forward(self, u_feat, m_feat, e_attr):
        v_u = self.user_embed(u_feat)
        v_m = self.movie_embed(m_feat)

        interaction_term = v_u * v_m
        attn_input = torch.cat([interaction_term, e_attr], dim=-1)

        attn_score = self.attn_layer(attn_input)

        weighted_interaction = interaction_term * attn_score
        weighted_interaction = self.dropout(weighted_interaction)
        interaction_sum = torch.sum(weighted_interaction, dim=1)

        linear_term = self.bias + self.linear_u(u_feat).squeeze(1) + self.linear_m(m_feat).squeeze(1)

        prediction = linear_term + interaction_sum

        return prediction


# ===================================================================
# 2. 定数・パス設定
# ===================================================================

PROJECT_ROOT = Path("/Users/watanabesaki/PycharmProjects/AFM")
DATA_PROCESSED = PROJECT_ROOT / "data/processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 💥 Testデータで最終評価を行うため、VAL_PTをTEST_PTに置き換えます
TEST_PT = DATA_PROCESSED / "hetero_graph_test.pt"
MODEL_LOAD_PATH = MODELS_DIR / "afm_model.pt"
OUTPUT_CSV_PATH = RESULTS_DIR / "afm_inference_errors_test_set.csv" # 💥 ファイル名変更

# 💥 修正: 最終的な学習時のパラメータに合わせる (K_EMBED=32, ATTN_SIZE=32)
K_EMBED = 32
ATTN_SIZE = 32
DROPOUT_RATE = 0.2

# u_feat の絶対インデックス
USER_MEAN_ABS_INDEX = 26
USER_STD_ABS_INDEX = 27
SLICED_USER_FEAT_DIM = 198 # μとσを除いた次元


# ===================================================================
# 3. データロード・デノーマライズロジック (198D対応)
# ===================================================================

def load_data_and_extract_info(path: Path):
    """グラフデータをロードし、デノーマライズパラメータを抽出する (198D対応)"""
    data = torch.load(path, weights_only=False)

    u_feat_all = data['user'].x # 200D
    m_feat_all = data['movie'].x

    edge_data = data['user', 'review', 'movie']

    u_indices, m_indices = edge_data.edge_index
    e_attr = edge_data.edge_attr
    y_norm = edge_data.y  # Z-Scoreターゲット

    # 1. RawなMean/Stdを抽出 (デノーマライズ用)
    user_mean_rating_all = u_feat_all[:, USER_MEAN_ABS_INDEX]
    user_std_rating_all = u_feat_all[:, USER_STD_ABS_INDEX]

    # 標準偏差が0の場合のゼロ除算防止 (1.0に置換)
    user_std_rating_all = torch.where(user_std_rating_all == 0, torch.tensor(1.0, dtype=torch.float32), user_std_rating_all)

    # 2. 💥 特徴量から Mean/Std を削除 (モデル入力用: 200D -> 198D)
    features_before_mu = u_feat_all[:, :USER_MEAN_ABS_INDEX]         # 0 to 25
    features_after_sigma = u_feat_all[:, USER_STD_ABS_INDEX + 1:]   # 28 to 199
    u_feat_sliced_all = torch.cat([features_before_mu, features_after_sigma], dim=1) # 198D

    # 3. エッジの数だけ特徴量とパラメータを複製
    u_feat_indexed = u_feat_sliced_all[u_indices] # 💥 198D
    m_feat_indexed = m_feat_all[m_indices]

    # Raw Ratingの真の値の計算: y_true_raw = (y_norm * u_std) + u_mean
    u_mean_indexed = user_mean_rating_all[u_indices]
    u_std_indexed = user_std_rating_all[u_indices]
    y_true_raw = (y_norm * u_std_indexed) + u_mean_indexed

    return u_feat_indexed, m_feat_indexed, e_attr, y_true_raw, u_indices, m_indices, u_mean_indexed, u_std_indexed


# ===================================================================
# 4. MAIN FUNCTION
# ===================================================================

def main():
    # 推論時もシード固定は維持
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 80)
    print("AFM MODEL INFERENCE AND EVALUATION (Test Set - Final Report)")
    print("=" * 80)

    # 1. データロード (TEST_PTを使用)
    u_feat_indexed, m_feat_indexed, e_attr_indexed, y_true_raw, u_indices, m_indices, u_mean_indexed, u_std_indexed = load_data_and_extract_info(
        TEST_PT) # 💥 TEST_PTを使用

    user_dim = u_feat_indexed.shape[1]
    movie_dim = m_feat_indexed.shape[1]
    edge_dim = e_attr_indexed.shape[1]

    # 2. モデル初期化と重みロード
    # user_dim=198 でモデルを初期化
    model = AFM(user_dim, movie_dim, edge_dim, embedding_dim=K_EMBED, attn_size=ATTN_SIZE, dropout_rate=DROPOUT_RATE)

    try:
        model.load_state_dict(torch.load(MODEL_LOAD_PATH))
        print(f"[Model] Loaded best model weights from {MODEL_LOAD_PATH.name} (User Dim: {user_dim}D)")
    except Exception as e:
        print(f"[ERROR] Failed to load model weights. Please ensure the model was trained with 198D features. Error: {e}")
        return

    # -------------------------------------------------------------
    # 💥 追加: レビュー特徴量 (e_attr) と真値 (y_true_raw) の相関分析
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("【DEBUG: レビュー特徴量 (e_attr) のデータリーク検証】")

    # テンソルをNumpyに変換
    e_attr_np = e_attr_indexed.numpy()
    y_true_raw_np = y_true_raw.numpy()

    # 相関を計算 (各特徴量次元 vs y_true_raw)
    correlations = []
    for i in range(e_attr_np.shape[1]):
        # e_attrの特徴量がすべて定数（分散が0）の場合、相関計算でNaNが出るためスキップ
        if np.std(e_attr_np[:, i]) > 1e-6:
            corr = np.corrcoef(e_attr_np[:, i], y_true_raw_np)[0, 1]
        else:
            corr = 0.0  # 変化しない特徴量はリークの可能性が低い
        correlations.append(corr)

    # 絶対値の相関でソート
    abs_correlations = np.abs(correlations)
    sorted_indices = np.argsort(abs_correlations)[::-1]

    # トップ5の相関を出力
    print("-" * 80)
    print("Top 5 E_ATTR Features Correlated with Y_TRUE_RAW (Raw Rating):")
    for i in sorted_indices[:5]:
        print(f"  Dim {i:2d}: Correlation R = {correlations[i]:.6f}")

    print("-" * 80)
    print(f"💥 警告: どの次元も相関の絶対値が 0.95 を超える場合、深刻なリークが疑われます。")
    print("=" * 80)
    # -------------------------------------------------------------

    # 3. 推論とデノーマライズ
    model.eval()
    with torch.no_grad():
        # ... (既存の推論ロジックは省略) ...
        y_pred_norm = model(u_feat_indexed, m_feat_indexed, e_attr_indexed)
        y_pred_raw = (y_pred_norm * u_std_indexed) + u_mean_indexed
        y_pred = torch.clamp(y_pred_raw, min=1.0, max=10.0)
        y_true_clamped = torch.clamp(y_true_raw, min=1.0, max=10.0)

        # RMSEとMAEの計算 (Raw Ratingスケール)
        rmse = torch.sqrt(F.mse_loss(y_pred, y_true_clamped)).item()
        mae = F.l1_loss(y_pred, y_true_clamped).item()

        # ... (既存のデバッグ表示とCSV出力は省略) ...
        print("-" * 30)
        print(f"[DEBUG] y_true_raw sample (clamped): {y_true_clamped[:5].numpy()}")
        print(f"[DEBUG] y_pred_raw sample (clamped): {y_pred[:5].numpy()}")
        print("-" * 30)
        print(f"[Accuracy Metrics] Final Test RMSE (Raw Rating): {rmse:.4f}")
        print(f"[Accuracy Metrics] Final Test MAE (Raw Rating): {mae:.4f}")
        print("-" * 30)

        # 誤差の一覧を作成
        errors = (y_pred - y_true_clamped).numpy()
        results_df = pd.DataFrame({
            'user_idx': u_indices.numpy(),
            'movie_idx': m_indices.numpy(),
            'y_true_raw': y_true_clamped.numpy(),
            'y_pred': y_pred.numpy(),
            'error': errors
        })

    results_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"[Output] Error list saved to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()