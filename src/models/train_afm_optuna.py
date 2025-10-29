# src/models/train_afm_optuna.py (Optunaによるハイパラ探索版)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import os
import optuna  # 💥 新規: Optunaをインポート


# ===================================================================
# 1. 安定化措置: ランダムシードの固定
# ===================================================================

def set_seed(seed_value=42):
    """すべてのランダムシードを固定し、学習の再現性を確保します。"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed_value)


# ===================================================================
# 2. AFM MODEL DEFINITION (Z-Score予測モデル) - train_afm.pyからそのまま流用
# ===================================================================
class AFM(nn.Module):
    def __init__(self, user_dim, movie_dim, edge_dim,
                 embedding_dim=32,
                 attn_size=32,
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
# 3. TRAIN HELPER FUNCTIONS
# ===================================================================

# === パスと固定設定 ===
PROJECT_ROOT = Path("/Users/watanabesaki/PycharmProjects/AFM")  # 💥 自身のプロジェクトルートに合わせてください
DATA_PROCESSED = PROJECT_ROOT / "data/processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

TRAIN_PT = DATA_PROCESSED / "hetero_graph_train.pt"
VAL_PT = DATA_PROCESSED / "hetero_graph_val.pt"
# 探索時はモデルを一時ファイルに保存するか、ベストなものだけ保存します
MODEL_SAVE_PATH = MODELS_DIR / "afm_model_optuna_best.pt"

# 固定パラメータ
NUM_EPOCHS = 100  # 探索時間を短縮するため、エポック数を減らすことを推奨
BATCH_SIZE = 512
DROPOUT_RATE = 0.2

# ユーザー/エッジ特徴量のリーク次元
USER_MEAN_ABS_INDEX = 26
USER_STD_ABS_INDEX = 27
EDGE_LEAK_START_INDEX = 36  # Dim 36, 37 を削除


def load_and_preprocess_data(path: Path):
    """グラフデータをロードし、AFM形式のテンソルに変換。デノーマライズに必要なパラメータも抽出。"""
    try:
        data = torch.load(path, weights_only=False)
    except FileNotFoundError:
        print(f"[ERROR] Data not found at {path}. Aborting.")
        return None, None, None, None, None, None, None

    u_feat_all = data['user'].x  # 200D
    m_feat_all = data['movie'].x

    # デノーマライズパラメータを絶対インデックスから抽出
    user_mean_rating = u_feat_all[:, USER_MEAN_ABS_INDEX]
    user_std_rating = u_feat_all[:, USER_STD_ABS_INDEX]
    user_std_rating = torch.where(user_std_rating == 0, torch.tensor(1.0, dtype=torch.float32), user_std_rating)

    edge_data = data['user', 'review', 'movie']
    u_indices, m_indices = edge_data.edge_index
    e_attr_full = edge_data.edge_attr  # 38D
    y_norm = edge_data.y  # Z-Scoreターゲット

    # 1. ユーザー特徴量からμとσを削除 (200D -> 198D)
    features_before_mu = u_feat_all[:, :USER_MEAN_ABS_INDEX]
    features_after_sigma = u_feat_all[:, USER_STD_ABS_INDEX + 1:]
    u_feat_sliced_all = torch.cat([features_before_mu, features_after_sigma], dim=1)  # 198D
    u_feat_indexed = u_feat_sliced_all[u_indices]

    # 2. レビュー特徴量からリーク次元 (インデックス36, 37) を削除 (38D -> 36D)
    e_attr_sliced = e_attr_full[:, :EDGE_LEAK_START_INDEX]

    m_feat_indexed = m_feat_all[m_indices]

    # デノーマライズパラメータをエッジごとに取得
    u_mean_indexed = user_mean_rating[u_indices]
    u_std_indexed = user_std_rating[u_indices]

    # Raw Ratingの真の値 (Y) を計算 - 訓練ループで重み付けに使用
    y_raw = (y_norm * u_std_indexed) + u_mean_indexed

    return u_feat_indexed, m_feat_indexed, e_attr_sliced, y_norm, y_raw, u_mean_indexed, u_std_indexed


# Raw Rating スケールでの評価関数
def evaluate_model(model, u_feat, m_feat, e_attr, y_true_raw, u_mean, u_std):
    """検証データでのMAEとRMSEを計算 (Raw Ratingスケール)"""
    model.eval()
    with torch.no_grad():
        # 予測値はZ-Scoreスケール
        y_pred_norm = model(u_feat, m_feat, e_attr)

        # Z-Score予測値をデノーマライズ
        y_pred_raw = (y_pred_norm * u_std) + u_mean

        # Raw Ratingの範囲にClamp (モデル外部で適用)
        y_pred_raw = torch.clamp(y_pred_raw, min=1.0, max=10.0)
        y_true_raw = torch.clamp(y_true_raw, min=1.0, max=10.0)

        # RMSE/MAE は Raw Ratingの誤差で計算
        rmse = torch.sqrt(F.mse_loss(y_pred_raw, y_true_raw)).item()
        mae = F.l1_loss(y_pred_raw, y_true_raw).item()

    return rmse, mae


# ===================================================================
# 4. OPTUNA OBJECTIVE FUNCTION (探索対象の関数)
# ===================================================================

def objective(trial: optuna.Trial):
    """
    Optunaが最適化する目的関数。検証RMSEを返す。
    """
    set_seed(42)  # トライアルごとの再現性を確保

    # 1. ハイパーパラメータのサンプリング
    # 💥 修正: Optunaを使ってハイパラを定義
    K_EMBED = trial.suggest_categorical('K_EMBED', [32, 64, 128])  # 埋め込み次元
    ATTN_SIZE = trial.suggest_categorical('ATTN_SIZE', [32, 64, 128])  # アテンション隠れ層サイズ
    LEARNING_RATE = trial.suggest_float('LEARNING_RATE', 1e-5, 1e-3, log=True)  # 学習率 (対数スケール)
    LOW_RATING_WEIGHT = trial.suggest_float('LOW_RATING_WEIGHT', 1.0, 30.0)  # 低評価重み

    print("-" * 80)
    print(
        f"Trial {trial.number}: K_EMBED={K_EMBED}, ATTN_SIZE={ATTN_SIZE}, LR={LEARNING_RATE:.6f}, WEIGHT={LOW_RATING_WEIGHT:.2f}")
    print("-" * 80)

    # 2. データロード (共通)
    u_feat_train, m_feat_train, e_attr_train, y_train_norm, y_train_raw, _, _ = load_and_preprocess_data(TRAIN_PT)
    u_feat_val_tmp, m_feat_val_tmp, e_attr_val_tmp, y_val_norm, y_val_raw, u_mean_val, u_std_val = load_and_preprocess_data(
        VAL_PT)

    if u_feat_train is None or u_feat_val_tmp is None:
        raise FileNotFoundError("Training or validation data not found.")

    # 3. モデル初期化
    user_dim = u_feat_train.shape[1]
    movie_dim = m_feat_train.shape[1]
    edge_dim = e_attr_train.shape[1]

    model = AFM(user_dim, movie_dim, edge_dim, embedding_dim=K_EMBED, attn_size=ATTN_SIZE, dropout_rate=DROPOUT_RATE)

    # 4. 訓練設定
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # 5. 訓練データローダー
    train_dataset = TensorDataset(u_feat_train, m_feat_train, e_attr_train, y_train_norm, y_train_raw)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 6. 訓練ループ
    best_val_rmse = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()

        # 早期終了 (Pruning) を導入
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        for u_feat, m_feat, e_attr, y_true_norm, y_true_raw in train_loader:
            optimizer.zero_grad()
            y_pred_norm = model(u_feat, m_feat, e_attr)

            # 重み付き損失の計算
            low_rating_mask = y_true_raw <= 4.0
            weights = torch.ones_like(y_true_raw, dtype=torch.float32)
            weights[low_rating_mask] = LOW_RATING_WEIGHT
            squared_error = (y_pred_norm - y_true_norm) ** 2
            loss = (weights * squared_error).mean()

            loss.backward()
            optimizer.step()

        # 検証
        val_rmse, _ = evaluate_model(model, u_feat_val_tmp, m_feat_val_tmp, e_attr_val_tmp, y_val_raw, u_mean_val,
                                     u_std_val)

        # Optunaに現在のスコアを報告
        trial.report(val_rmse, epoch)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            # ベストモデルを保存することも可能だが、ここでは速度優先でスキップ

    # 7. 最終的な最小検証RMSEを返す (Optunaの目的は最小化)
    return best_val_rmse


# ===================================================================
# 5. OPTUNA STUDY MAIN
# ===================================================================

def main_optuna():
    set_seed(42)
    print("=" * 80)
    print("AFM OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)

    # データベースを設定して結果を永続化することを推奨 (例: sqlite:///db.sqlite3)
    # 永続化しない場合は `storage=None` でメモリ上で実行されます
    study = optuna.create_study(
        direction='minimize',  # 最小化する指標 (RMSE)
        sampler=optuna.samplers.TPESampler(seed=42),  # 探索アルゴリズム
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)  # 早期終了の設定
    )

    # 探索の実行
    # n_trials: 試行回数 (探索の深さ/広さ)
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    # 結果の表示
    print("\n[Complete] Optimization finished.")
    print("-" * 60)
    print(f"Best Trial (RMSE={study.best_value:.4f})")
    print("-" * 60)
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # 最適なハイパラを使ってモデルを再学習・保存する処理を追加することもできます

    # 💥 ベストなハイパラを `train_afm.py` に反映させて、改めてエポックを増やして学習を実行してください。


if __name__ == "__main__":
    main_optuna()