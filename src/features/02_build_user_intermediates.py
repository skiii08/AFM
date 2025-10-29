# src/features/02_build_user_intermediates.py

import pandas as pd
import numpy as np
import json
import torch
from pathlib import Path
from itertools import combinations
from collections import defaultdict
from tqdm import tqdm

# === パス設定 ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data/raw"
DATA_PROCESSED = PROJECT_ROOT / "data/processed"
FEATURES_DIR = DATA_PROCESSED / "features"

# ディレクトリ作成
DATA_PROCESSED.mkdir(exist_ok=True)
FEATURES_DIR.mkdir(exist_ok=True)

# 入力
MOVIES_CSV = DATA_RAW / "movies_metadata.csv"
REVIEWS_CSV = DATA_RAW / "reviews.csv"

# 出力
USER_ACTOR_JSON = FEATURES_DIR / "user_actor_preference.json"
USER_DIRECTOR_JSON = FEATURES_DIR / "user_director_preference.json"
USER_GENRE_PAIR_PREF_JSON = FEATURES_DIR / "user_genre_pair_preference.json"
GENRE_PAIR_VOCAB_JSON = FEATURES_DIR / "genre_pair_vocab.json"


# --- 補助関数 (パーソン/ジャンル) ---

def parse_list_cell(s: str) -> list[str]:
    """区切り文字(;)でリストをパース"""
    if pd.isna(s) or s == "":
        return []
    return [x.strip() for x in str(s).split(";") if x.strip()]


def compute_user_person_stats(reviews_df, movies_df, global_mean, person_type="actor"):
    """ユーザー×人物の評価統計を計算 (ベイジアン平滑化適用)"""
    person_col = 'actors' if person_type == 'actor' else 'directors'

    # 💥 新規: ベイジアン平滑化の信頼度重み C
    C_WEIGHT = 5.0  # この値はデータセットのサイズに応じて調整可能

    # 映画とレビューを結合
    df = reviews_df.merge(movies_df[['movie_id', person_col]], on='movie_id', how='left')

    user_person_stats = defaultdict(lambda: defaultdict(lambda: {'ratings': [], 'count': 0}))

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Calculating {person_type} stats"):
        persons = parse_list_cell(row[person_col])
        user = row['user_name']
        rating = row['rating_raw']

        for person in persons:
            user_person_stats[user][person]['ratings'].append(rating)
            user_person_stats[user][person]['count'] += 1

    # 平均評価を計算
    user_top_persons = defaultdict(list)
    for user, persons in user_person_stats.items():
        for person, data in persons.items():
            N = data['count']
            if N > 0:
                Mean_item = np.mean(data['ratings'])

                # 💥 修正: ベイジアンアベレージングを適用
                # Rating_smooth = (C * Mean_global + N * Mean_item) / (C + N)
                avg_rating_smooth = (C_WEIGHT * global_mean + N * Mean_item) / (C_WEIGHT + N)

                user_top_persons[user].append({
                    'name': person,
                    'avg_rating': avg_rating_smooth,  # 平滑化された評価を使用
                    'count': N
                })
        # 評価数が多い順にソート (Top-K抽出は03で実施)
        user_top_persons[user].sort(key=lambda x: x['count'], reverse=True)

    return user_top_persons


def compute_genre_pair_preference(reviews_df, movies_df):
    """ユーザーのジャンルペア嗜好を計算 (171D)"""

    # 映画とレビューを結合
    df = reviews_df.merge(movies_df, on='movie_id', how='left')

    genre_cols = [c for c in movies_df.columns if c.startswith('genre_')]

    # ジャンルリスト抽出
    def get_genres(row):
        return [c.replace('genre_', '') for c in genre_cols if row[c] == 1]

    # movies_metadataにはgenre_XXX列がない可能性があるため、`genres`列から取得
    if not genre_cols:
        all_genres = set()
        df['genres'].dropna().apply(lambda x: all_genres.update(parse_list_cell(x)))
        all_genres = sorted(list(all_genres))
        # genres列をMulti-hotに変換
        for g in all_genres:
            df[f'genre_{g}'] = df['genres'].apply(lambda x: 1 if g in parse_list_cell(x) else 0)
        genre_cols = [f'genre_{g}' for g in all_genres]

    # ジャンルペアのボキャブラリを生成 (171ペア)
    all_genres_names = [c.replace('genre_', '') for c in genre_cols]
    genre_pairs = sorted(list(combinations(all_genres_names, 2)))
    genre_pair_vocab = {pair: idx for idx, pair in enumerate(genre_pairs)}

    user_genre_pair_pref = defaultdict(lambda: defaultdict(lambda: {'ratings': [], 'count': 0}))

    # ユーザーごとのペア嗜好を集計
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Calculating genre pair stats"):
        user = row['user_name']
        rating = row['rating_raw']

        # 映画のジャンルを取得
        present_genres = [g for g in all_genres_names if row[f'genre_{g}'] == 1]

        for pair in combinations(present_genres, 2):
            sorted_pair = tuple(sorted(pair))
            if sorted_pair in genre_pair_vocab:
                user_genre_pair_pref[user][sorted_pair]['ratings'].append(rating)
                user_genre_pair_pref[user][sorted_pair]['count'] += 1

    # 平均評価と分散を計算
    user_final_pref = defaultdict(lambda: {})
    for user, pairs in user_genre_pair_pref.items():
        for pair, data in pairs.items():
            if data['count'] >= 5:  # 最低5レビューを要件とする
                user_final_pref[user][f"{pair[0]}×{pair[1]}"] = {
                    'avg_rating': float(np.mean(data['ratings'])),
                    'std': float(np.std(data['ratings'])),
                    'count': data['count']
                }

    # pairのvocabを文字列に変換して保存
    genre_pair_vocab_str = {f"{p[0]}×{p[1]}": idx for p, idx in genre_pair_vocab.items()}

    return user_final_pref, genre_pair_vocab_str


# --- メイン処理 ---
def main():
    print("=" * 80)
    print("02_BUILD_USER_INTERMEDIATES (A/D & Genre Pair Prefs)")
    print("=" * 80)

    # データ読み込み
    movies_df = pd.read_csv(MOVIES_CSV)
    reviews_df = pd.read_csv(REVIEWS_CSV)

    # 💥 新規: 全レビューの平均値を計算 (ベイジアン平滑化に利用)
    global_mean_rating = reviews_df['rating_raw'].mean()
    print(f"[Info] Global Mean Rating: {global_mean_rating:.4f}")

    # 1. 俳優/監督の嗜好
    # 💥 修正: global_mean_rating を引数に追加
    user_top_actors = compute_user_person_stats(reviews_df, movies_df, global_mean_rating, "actor")
    user_top_directors = compute_user_person_stats(reviews_df, movies_df, global_mean_rating, "director")

    # 2. ジャンルペアの嗜好
    user_genre_pair_pref, genre_pair_vocab_str = compute_genre_pair_preference(reviews_df, movies_df)

    # 3. 保存
    print("\n[Save] Saving intermediate features...")

    # 俳優
    with open(USER_ACTOR_JSON, 'w') as f:
        json.dump(user_top_actors, f, indent=2)
    print(f"  Saved Actor Prefs: {USER_ACTOR_JSON}")

    # 監督
    with open(USER_DIRECTOR_JSON, 'w') as f:
        json.dump(user_top_directors, f, indent=2)
    print(f"  Saved Director Prefs: {USER_DIRECTOR_JSON}")

    # ジャンルペア嗜好
    with open(USER_GENRE_PAIR_PREF_JSON, 'w') as f:
        json.dump(user_genre_pair_pref, f, indent=2)
    print(f"  Saved Genre Pair Prefs: {USER_GENRE_PAIR_PREF_JSON}")

    # ジャンルペアVocab
    with open(GENRE_PAIR_VOCAB_JSON, 'w') as f:
        json.dump(genre_pair_vocab_str, f, indent=2)
    print(f"  Saved Genre Pair Vocab: {GENRE_PAIR_VOCAB_JSON}")


if __name__ == "__main__":
    main()