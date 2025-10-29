# src/analysis/analyze_rating_integrity.py (または analyze_rating.py)

import pandas as pd
import numpy as np
from pathlib import Path

# === 設定 (ユーザーの環境に合わせてパスを設定) ===
PROJECT_ROOT = Path("/Users/watanabesaki/PycharmProjects/AFM")
DATA_RAW = PROJECT_ROOT / "data/raw"
REVIEWS_CSV = DATA_RAW / "reviews.csv"
LOW_RATING_THRESHOLD = 4.0


def analyze_user_statistics_integrity():
    """レビュー数が少ないユーザーの統計情報 (NaN/0 問題) を分析する"""
    print("=" * 80)
    print("🔬 USER STATISTICS INTEGRITY CHECK (Finding the 0.0 Root Cause)")
    print("=" * 80)

    if not REVIEWS_CSV.exists():
        print(f"[ERROR] Reviews CSV not found: {REVIEWS_CSV}")
        return

    # DataFrameをロード
    reviews_df = pd.read_csv(REVIEWS_CSV)

    # ユーザーごとの統計情報を集計
    user_stats = reviews_df.groupby('user_name').agg(
        review_count=('rating_raw', 'count'),
        mean_rating=('rating_raw', 'mean'),
        std_rating=('rating_raw', 'std'),
        low_rating_count=('rating_raw', lambda x: (x <= LOW_RATING_THRESHOLD).sum())
    ).reset_index()

    print(f"Total Unique Users: {len(user_stats)}")

    # --------------------------------------------------
    # 1. レビュー数と統計値の欠損分析
    # --------------------------------------------------
    print("\n--- 1. Review Count Distribution ---")
    review_counts = user_stats['review_count'].value_counts().sort_index()

    # 💥 修正: to_markdown() を to_string() に変更
    print(review_counts.head(10).to_string())

    # --------------------------------------------------
    # 2. NaNが発生するユーザーの特定 (レビュー数 = 1)
    # --------------------------------------------------
    # レビュー数1件のユーザーは、標準偏差(std_rating)がNaNになる
    users_with_nan_std = user_stats[user_stats['std_rating'].isna()]

    print("\n--- 2. Users Causing NaN in Standard Deviation ---")
    print(f"Users with NaN std_rating (N=1): {len(users_with_nan_std)}")

    if len(users_with_nan_std) > 0:
        mean_rating_of_n1 = users_with_nan_std['mean_rating'].mean()
        print(f"  These {len(users_with_nan_std)} users have a mean_rating of: {mean_rating_of_n1:.4f}")

    # --------------------------------------------------
    # 3. 統計値が0であるユーザーの特定 (データ処理バグのターゲット)
    # --------------------------------------------------
    users_with_zero_mean = user_stats[user_stats['mean_rating'] == 0]

    print("\n--- 3. Users with Mean Rating exactly 0.0 (Data Loss Suspected) ---")
    print(f"Users with mean_rating = 0.0: {len(users_with_zero_mean)}")


if __name__ == "__main__":
    analyze_user_statistics_integrity()