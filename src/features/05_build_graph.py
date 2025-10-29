import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path
from torch_geometric.data import HeteroData

# Config Import
from src.models.config import EDGE_FEAT_DIM

# === パス設定 ===
# 実行ファイル(05) -> features(1) -> src(2) -> AFM(3)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data/raw"
DATA_PROCESSED = PROJECT_ROOT / "data/processed"
FEATURES_DIR = DATA_PROCESSED / "features"

# 入力
REVIEWS_CSV = DATA_RAW / "reviews.csv"
USER_FEATURES_PT = DATA_PROCESSED / "user_features.pt"
MOVIE_FEATURES_PT = DATA_PROCESSED / "movie_features.pt"
USER_FEATURE_INFO_JSON = FEATURES_DIR / "user_feature_info.json"
MOVIE_FEATURE_INFO_JSON = FEATURES_DIR / "movie_feature_info.json"

# 出力
TRAIN_PT = DATA_PROCESSED / "hetero_graph_train.pt"
VAL_PT = DATA_PROCESSED / "hetero_graph_val.pt"
TEST_PT = DATA_PROCESSED / "hetero_graph_test.pt"

# エッジ特徴量カラムの決定 (38Dを想定)
# ユーザー特徴量構築時のアスペクト特徴量と同じものを使用します
ASPECT_COLUMNS = [
    'mention_acting_performance', 'mention_artistic_design', 'mention_audio_music',
    'mention_casting_choices', 'mention_character_development', 'mention_commercial_context',
    'mention_comparative_analysis', 'mention_editing_pacing', 'mention_emotion',
    'mention_expectation', 'mention_filmmaking_direction', 'mention_genre_style',
    'mention_recommendation', 'mention_story_plot', 'mention_technical_visuals',
    'mention_themes_messages', 'mention_viewing_experience', 'mention_writing_dialogue',
    'sentiment_acting_performance', 'sentiment_artistic_design', 'sentiment_audio_music',
    'sentiment_casting_choices', 'sentiment_character_development', 'sentiment_commercial_context',
    'sentiment_comparative_analysis', 'sentiment_editing_pacing', 'sentiment_emotion',
    'sentiment_expectation', 'sentiment_filmmaking_direction', 'sentiment_genre_style',
    'sentiment_recommendation', 'sentiment_story_plot', 'sentiment_technical_visuals',
    'sentiment_themes_messages', 'sentiment_viewing_experience', 'sentiment_writing_dialogue',
    # 36D + 2D: 例としてユーザーの正規化評価と、映画の正規化評価(rating_raw)を加える
    'rating_user_norm', 'rating_raw'
]

# 実際の次元が38Dであることを確認
if len(ASPECT_COLUMNS) != EDGE_FEAT_DIM:
    print(f"[WARNING] Aspect column count mismatch: Expected {EDGE_FEAT_DIM}D, got {len(ASPECT_COLUMNS)}D.")


def build_graph(reviews_subset: pd.DataFrame, user_to_idx, movie_to_idx, user_features, movie_features) -> HeteroData:
    """レビューのサブセットからHeteroDataグラフを構築"""
    if reviews_subset.empty:
        # エッジがない場合は、ノード特徴量だけを持つ空のグラフを返す
        data = HeteroData()
        data['user'].x = user_features
        data['movie'].x = movie_features
        return data

    # ユーザー/映画のインデックスリストを作成
    # reviews_subset['user_name']は、user_to_idxに含まれることが保証されている
    user_indices = [user_to_idx[uid] for uid in reviews_subset['user_name']]
    movie_indices = [movie_to_idx[mid] for mid in reviews_subset['movie_id']]

    data = HeteroData()

    # ノード特徴量を設定
    data['user'].x = user_features
    data['movie'].x = movie_features

    # エッジ（ユーザー→レビュー→映画）
    edge_index = torch.tensor([user_indices, movie_indices], dtype=torch.long)
    data['user', 'review', 'movie'].edge_index = edge_index

    # エッジ属性（38D）
    # 必要なカラムがレビューDFに存在するか確認
    missing_cols = [c for c in ASPECT_COLUMNS if c not in reviews_subset.columns]
    if missing_cols:
        print(f"[ERROR] Missing edge feature columns: {missing_cols}. Using 0D edge features.")
        edge_attr = torch.zeros((len(reviews_subset), 0), dtype=torch.float32)
    else:
        edge_attr = torch.tensor(reviews_subset[ASPECT_COLUMNS].values, dtype=torch.float32)
        # エッジ特徴量の次元が38Dであることを確認
        if edge_attr.size(1) != EDGE_FEAT_DIM:
            print(f"[WARNING] Edge attribute dimension mismatch: Expected {EDGE_FEAT_DIM}D, got {edge_attr.size(1)}D.")

    data['user', 'review', 'movie'].edge_attr = edge_attr

    # ラベル（ターゲット値: 正規化済み評価）
    labels = torch.tensor(reviews_subset['rating_user_norm'].values, dtype=torch.float32)
    data['user', 'review', 'movie'].y = labels

    return data


def main():
    print("=" * 80)
    print("05_BUILD_GRAPH (Edge Feature 38D FINAL)")
    print("=" * 80)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # 1. ユーザー/映画インデックス情報の読み込み
    try:
        with open(USER_FEATURE_INFO_JSON, 'r') as f:
            user_info = json.load(f)
        with open(MOVIE_FEATURE_INFO_JSON, 'r') as f:
            movie_info = json.load(f)
    except FileNotFoundError:
        print("[ERROR] Feature info (user_feature_info.json or movie_feature_info.json) not found.")
        print("        Please ensure 03_build_user_features.py and 04_build_movie_features.py ran successfully.")
        return

    # 2. 特徴量テンソルの読み込み
    try:
        user_features = torch.load(USER_FEATURES_PT, weights_only=False)
        movie_features = torch.load(MOVIE_FEATURES_PT, weights_only=False)
    except FileNotFoundError as e:
        print(f"[ERROR] Feature tensor not found: {e.filename}. Aborting.")
        return

    # 💥 CRITICAL FIX: user_features.ptに格納されているユーザーのみでインデックスマップを再構築する
    # user_featuresの行数（840）と、user_info['user_names']の対応をチェックし、
    # ユーザー特徴量を持つユーザー名だけを抽出して新しいインデックスマップを作成する。

    # user_features_info.jsonのuser_namesは特徴量のインデックスに対応していると仮定
    expected_num_users = user_features.size(0)

    if len(user_info['user_names']) != expected_num_users:
        # 特徴量を持つユーザー名リストを、テンソルのサイズに合わせてスライス/または信頼する
        # ここでは、特徴量テンソルのサイズに合わせてユーザー名をスライスします
        print(
            f"[INFO] Adjusting user list from {len(user_info['user_names'])} to {expected_num_users} based on loaded features.")
        valid_user_names = user_info['user_names'][:expected_num_users]
    else:
        valid_user_names = user_info['user_names']

    # 新しいインデックスマップを作成
    user_to_idx = {name: i for i, name in enumerate(valid_user_names)}

    # movie_info['movie_ids']は整数リストなので、キーも整数になる
    movie_to_idx = {mid: i for i, mid in enumerate(movie_info['movie_ids'])}

    print(f"\n[Feature] Computing new {EDGE_FEAT_DIM}D Edge Features...")
    print(f"  Loaded {len(user_to_idx)} valid users and {len(movie_to_idx)} movies.")
    print(f"  User Feature Tensor size: {user_features.size(0)} (Matches valid user count: {len(user_to_idx)})")

    # 3. レビューデータの読み込みと分割
    try:
        reviews_df = pd.read_csv(REVIEWS_CSV)
        # 映画IDを整数に変換する
        reviews_df['movie_id'] = pd.to_numeric(reviews_df['movie_id'], errors='coerce').fillna(-1).astype(int)
        reviews_df = reviews_df[reviews_df['movie_id'] != -1].copy()

    except FileNotFoundError:
        print(f"[ERROR] Reviews file not found at {REVIEWS_CSV}. Aborting.")
        return

    # 'split'カラムの文字列を小文字に変換
    if 'split' not in reviews_df.columns:
        print("[ERROR] 'split' column not found in reviews.csv. Aborting graph building.")
        return

    reviews_df['split'] = reviews_df['split'].astype(str).str.lower()

    # Train/Val/Test に分割
    train_reviews = reviews_df[reviews_df['split'] == 'train'].copy()
    val_reviews = reviews_df[reviews_df['split'] == 'val'].copy()
    test_reviews = reviews_df[reviews_df['split'] == 'test'].copy()

    print(
        f"  Train/Val/Test split counts (before filtering): {len(reviews_df[reviews_df['split'] == 'train'])}/{len(reviews_df[reviews_df['split'] == 'val'])}/{len(reviews_df[reviews_df['split'] == 'test'])}")

    # 4. グラフ構築
    print("\n[Build] Building graphs...")

    # 💥 CRITICAL FIX: インデックスマップに存在しないデータを除外
    # このステップにより、エッジインデックスは user_to_idx のローカルインデックス（0〜839）に限定されます
    train_reviews = train_reviews[
        train_reviews['user_name'].isin(user_to_idx.keys()) & train_reviews['movie_id'].isin(movie_to_idx.keys())]
    val_reviews = val_reviews[
        val_reviews['user_name'].isin(user_to_idx.keys()) & val_reviews['movie_id'].isin(movie_to_idx.keys())]
    test_reviews = test_reviews[
        test_reviews['user_name'].isin(user_to_idx.keys()) & test_reviews['movie_id'].isin(movie_to_idx.keys())]

    train_graph = build_graph(train_reviews, user_to_idx, movie_to_idx, user_features, movie_features)
    val_graph = build_graph(val_reviews, user_to_idx, movie_to_idx, user_features, movie_features)
    test_graph = build_graph(test_reviews, user_to_idx, movie_to_idx, user_features, movie_features)

    # 5. 保存
    print("\n[Save] Saving graphs...")
    torch.save(train_graph, TRAIN_PT)
    torch.save(val_graph, VAL_PT)
    torch.save(test_graph, TEST_PT)
    print(f"  Graphs saved to {DATA_PROCESSED.name}/.")

    # 6. 最終結果の表示
    train_edges = train_graph[('user', 'review', 'movie')].edge_index.size(1) if ('user', 'review',
                                                                                  'movie') in train_graph.edge_types else 0
    val_edges = val_graph[('user', 'review', 'movie')].edge_index.size(1) if ('user', 'review',
                                                                              'movie') in val_graph.edge_types else 0
    test_edges = test_graph[('user', 'review', 'movie')].edge_index.size(1) if ('user', 'review',
                                                                                'movie') in test_graph.edge_types else 0

    print(f"  Train Edges: {train_edges}")
    print(f"  Validation Edges: {val_edges}")
    print(f"  Test Edges: {test_edges}")


if __name__ == "__main__":
    main()
