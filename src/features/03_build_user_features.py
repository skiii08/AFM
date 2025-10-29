# src/features/03_build_user_features.py (低評価信号強化版)

import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm
import sys

# Config Import
try:
    from src.models.config import USER_FEAT_DIM
except ImportError:
    USER_FEAT_DIM = 221

# === パス設定 (省略) ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data/raw"
DATA_PROCESSED = PROJECT_ROOT / "data/processed"
FEATURES_DIR = DATA_PROCESSED / "features"

USER_ACTOR_JSON = FEATURES_DIR / "user_actor_preference.json"
USER_DIRECTOR_JSON = FEATURES_DIR / "user_director_preference.json"
USER_GENRE_PAIR_JSON = FEATURES_DIR / "user_genre_pair_preference.json"
GENRE_PAIR_VOCAB_JSON = FEATURES_DIR / "genre_pair_vocab.json"
REVIEWS_CSV = DATA_RAW / "reviews.csv"
USER_FEATURES_PT = DATA_PROCESSED / "user_features.pt"


# --- 補助関数: コア統計情報計算 (低評価ロジックを追加) ---
def compute_core_stats(reviews_df: pd.DataFrame) -> Dict[str, Any]:
    """
    ユーザーごとの生の統計情報（mean, std, count）と低評価（<= 4.0）統計を計算する。
    """

    print("\n--- [DEBUG: Core Stats] Calculating Core User Statistics and Low Rating Features ---")

    # 低評価（<= 4.0）の件数を計算
    low_rating_df = reviews_df[reviews_df['rating_raw'] <= 4.0]
    low_rating_count = low_rating_df.groupby('user_name')['rating_raw'].count().rename('low_rating_count')

    # 生の統計情報を集計
    user_stats = reviews_df.groupby('user_name').agg(
        review_count=('rating_raw', 'count'),
        mean_rating=('rating_raw', 'mean'),
        std_rating=('rating_raw', 'std'),
    )

    # 低評価件数を結合し、NaNを0で埋める
    user_stats = user_stats.join(low_rating_count).fillna(0).reset_index()

    # 低評価比率を計算
    user_stats['low_rating_ratio'] = user_stats['low_rating_count'] / user_stats['review_count']

    user_info = {}

    # Z-score計算のためのグローバル統計値
    global_mean_count = user_stats['review_count'].mean()
    global_std_count = user_stats['review_count'].std()

    # 💥 CRITICAL: 低評価特徴量のグローバル統計
    global_mean_low_count = user_stats['low_rating_count'].mean()
    global_std_low_count = user_stats['low_rating_count'].std()
    global_mean_low_ratio = user_stats['low_rating_ratio'].mean()
    global_std_low_ratio = user_stats['low_rating_ratio'].std()

    print(f"  [DEBUG] Global Low Rating Count Mean: {global_mean_low_count:.4f}, Std: {global_std_low_count:.4f}")

    def z_score(value, mean, std):
        if std == 0 or pd.isna(std): return 0.0
        return (value - mean) / std

    for _, row in user_stats.iterrows():
        user_name = row['user_name']

        data = {
            'review_count_norm': z_score(row['review_count'], global_mean_count, global_std_count),
            'mean_rating': float(row['mean_rating']),
            'std_rating': float(row['std_rating']),
            'activity_score': 0.0,  # 未実装のため0.0

            # 💥 CRITICAL FIX: 低評価特徴量のZ-score化
            'low_rating_count_norm': z_score(row['low_rating_count'], global_mean_low_count, global_std_low_count),
            'low_rating_ratio_norm': z_score(row['low_rating_ratio'], global_mean_low_ratio, global_std_low_ratio),
        }

        user_info[user_name] = data

    # デバッグ出力
    sample_users = list(user_info.keys())[:3]
    for name in sample_users:
        data = user_info[name]
        print(
            f"  [DEBUG] Sample User: {name} | Mean: {data['mean_rating']:.4f}, LowCountNorm: {data['low_rating_count_norm']:.4f}, LowRatioNorm: {data['low_rating_ratio_norm']:.4f}")

    return user_info


# --- 補助関数 (パーソン/ジャンル) --- (変更なし: 省略)
def compute_person_features(top_persons: List[Dict[str, Any]]) -> List[float]:
    if not top_persons: return [0.0] * 4
    ratings = [p['avg_rating'] for p in top_persons]
    counts = [p['count'] for p in top_persons]
    return [float(np.mean(ratings)), float(np.log1p(np.sum(counts))), float(np.max(ratings)), float(np.min(ratings))]


def safe_get_raw_stat(user_data: Dict[str, Any], key: str, user_name: str):
    value = user_data.get(key)
    if value is None or pd.isna(value) or value == 0.0:
        if value != 0.0:  # 0.0の場合はレビュー数1件などでstdがNaNになるケース
            tqdm.write(
                f"🚨 CRITICAL WARNING (User Name: {user_name}, Key: {key}): 生の統計値が不正な値 ({value}/NaN) です。このユーザーはスキップされます。")
        return 0.0
    return float(value)


# --- メインの特徴量構築関数 (変更なし: 省略) ---
def build_user_features(user_names: List[str],
                        user_info: Dict[str, Any],
                        user_actor_pref: Dict[str, Any],
                        user_director_pref: Dict[str, Any],
                        user_genre_pair_pref: Dict[str, Any],
                        genre_pair_vocab: Dict[str, int],
                        ) -> torch.Tensor:
    final_features = []
    genre_pair_dim = len(genre_pair_vocab)
    ASPECT_DIM = 36

    for user_name in tqdm(user_names, desc="Building User Features"):
        user_feat = []
        if user_name not in user_info: continue
        user_data = user_info[user_name]

        # --- 1. コア統計特徴量 (6D) ---
        user_feat.append(user_data.get('review_count_norm', 0.0))  # Index 0

        user_mean_raw = safe_get_raw_stat(user_data, 'mean_rating', user_name)
        user_std_raw = safe_get_raw_stat(user_data, 'std_rating', user_name)

        if len(final_features) < 3:
            tqdm.write(
                f"  [DEBUG: Stats Check] User: {user_name} | Mean_Raw: {user_mean_raw:.4f}, Std_Raw: {user_std_raw:.4f}, LowC: {user_data.get('low_rating_count_norm'):.4f}")

        if user_mean_raw == 0.0 or user_std_raw == 0.0:
            continue

        user_feat.append(user_mean_raw)  # Index 1
        user_feat.append(user_std_raw)  # Index 2

        user_feat.append(user_data.get('activity_score', 0.0))  # Index 3
        user_feat.append(user_data.get('low_rating_count_norm', 0.0))  # 💥 Index 4: Low Rating Count
        user_feat.append(user_data.get('low_rating_ratio_norm', 0.0))  # 💥 Index 5: Low Rating Ratio

        # --- 2. トピック/アスペクトの嗜好 (36D) ---
        user_feat.extend([0.0] * ASPECT_DIM)

        # --- 3. 俳優/監督の嗜好 (8D) ---
        actor_feat = compute_person_features(user_actor_pref.get(user_name, []))
        user_feat.extend(actor_feat)

        director_feat = compute_person_features(user_director_pref.get(user_name, []))
        user_feat.extend(director_feat)

        # --- 4. ジャンルペアの嗜好 (171D) --- (省略)
        user_genre_data = user_genre_pair_pref.get(user_name, {})
        genre_pair_feat = np.zeros(genre_pair_dim, dtype=np.float32)

        for pair_str, pref_data in user_genre_data.items():
            if pair_str in genre_pair_vocab:
                idx = genre_pair_vocab[pair_str]
                genre_pair_feat[idx] = pref_data.get('avg_rating', 0.0)
        user_feat.extend(genre_pair_feat.tolist())

        final_features.append(user_feat)

        if len(final_features) <= 3:
            tqdm.write(
                f"  [DEBUG: Final Vector] User: {user_name} | D={len(user_feat)} | LowC_Norm (Idx 4): {user_feat[4]:.4f}, LowR_Norm (Idx 5): {user_feat[5]:.4f}")

    user_features_final = torch.tensor(final_features, dtype=torch.float32)

    if user_features_final.shape[1] != USER_FEAT_DIM:
        tqdm.write(f"🚨 CRITICAL DIMENSION MISMATCH: Expected {USER_FEAT_DIM}D, got {user_features_final.shape[1]}D.")

    return user_features_final


# --- メイン処理 (省略) ---
def main():
    print("=" * 80)
    print("03_BUILD_USER_FEATURES (DEBUG MODE: Low Rating Signal FIXED)")
    print("=" * 80)

    reviews_df = pd.read_csv(REVIEWS_CSV)

    # 中間ファイル読み込み (省略)
    try:
        with open(USER_ACTOR_JSON, 'r') as f:
            user_actor_pref = json.load(f)
        with open(USER_DIRECTOR_JSON, 'r') as f:
            user_director_pref = json.load(f)
        with open(USER_GENRE_PAIR_JSON, 'r') as f:
            user_genre_pair_pref = json.load(f)
        with open(GENRE_PAIR_VOCAB_JSON, 'r') as f:
            genre_pair_vocab = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ エラー: 中間ファイルが見つかりません。02_build_user_intermediates.py を実行してください: {e}")
        return

    user_names = reviews_df['user_name'].unique().tolist()
    user_info = compute_core_stats(reviews_df)

    user_features_final = build_user_features(
        user_names, user_info, user_actor_pref, user_director_pref,
        user_genre_pair_pref, genre_pair_vocab
    )

    print(
        f"\n[Save] Saving final user features (N={user_features_final.shape[0]}, D={user_features_final.shape[1]})...")
    torch.save(user_features_final, USER_FEATURES_PT)
    print(f"  Features saved to: {USER_FEATURES_PT}")


if __name__ == "__main__":
    main()