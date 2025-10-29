import torch
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import re

# === 設定 (ここを実際のファイルパスに合わせてください) ===
PROJECT_ROOT = Path("/Users/watanabesaki/PycharmProjects/AFM")
DATA_PROCESSED = PROJECT_ROOT / "data/processed"
FEATURES_DIR = DATA_PROCESSED / "features"
# テストグラフを分析
GRAPH_FILE = DATA_PROCESSED / "hetero_graph_test.pt"
GENRE_PAIR_VOCAB_JSON = FEATURES_DIR / "genre_pair_vocab.json"

# --- 定数: 最終的な特徴量次元 ---
USER_FEAT_DIM_FINAL = 221
MOVIE_FEAT_DIM_FINAL = 636
EDGE_FEAT_DIM_FINAL = 38

# --- Edge Feature Names (38D) ---
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
    'rating_user_norm', 'rating_raw'  # 36D + 2D = 38D
]


# --- ユーザー特徴量名生成 (221D) ---

def load_genre_pair_names() -> List[str]:
    """genre_pair_vocab.jsonからジャンルペア名をロード (171Dを想定)"""
    expected_dim = 171
    try:
        with open(GENRE_PAIR_VOCAB_JSON, 'r') as f:
            vocab = json.load(f)

        # 値(インデックス)でソートし、キー(ジャンル名)のリストを取得
        sorted_items = sorted(vocab.items(), key=lambda item: item[1])
        names = [item[0] for item in sorted_items]

        if len(names) != expected_dim:
            print(
                f"[WARNING] ジャンルペアの語彙数が{expected_dim}個ではありません ({len(names)}個)。"
            )
        return names

    except FileNotFoundError:
        print(f"[ERROR] ジャンルペア語彙ファイルが見つかりません: {GENRE_PAIR_VOCAB_JSON}")
        return [f"genre_pair_pref_{i:03d}" for i in range(expected_dim)]


def get_user_feature_names() -> Tuple[List[str], int, int]:
    """221Dのユーザー特徴量の項目名を構築し、mean/stdのインデックスを返す"""

    # 1. コア統計値 (6D)
    stat_names = [
        "user_review_count_norm",  # Index 0
        "user_mean_rating_norm",  # Index 1 👈 これが非正規化に使用される平均μ
        "user_std_rating_norm",  # Index 2 👈 これが非正規化に使用される標準偏差σ
        "user_activity_score",  # Index 3
        "user_low_rating_count_norm",  # Index 4 (新規)
        "user_low_rating_ratio_norm"  # Index 5 (新規)
    ]

    # 2. アスペクト嗜好 (36D)
    aspect_bases = [
        'acting_performance', 'artistic_design', 'audio_music', 'casting_choices',
        'character_development', 'commercial_context', 'comparative_analysis',
        'editing_pacing', 'emotion', 'expectation', 'filmmaking_direction',
        'genre_style', 'recommendation', 'story_plot', 'technical_visuals',
        'themes_messages', 'viewing_experience', 'writing_dialogue'
    ]
    aspect_names = []
    aspect_names.extend([f"zscore_{a}" for a in aspect_bases])
    aspect_names.extend([f"sentiment_{a}" for a in aspect_bases])

    # 3. 俳優/監督の嗜好 (8D)
    ad_names = []
    ad_names.extend([f"actor_pref_top{i + 1}" for i in range(4)])
    ad_names.extend([f"director_pref_top{i + 1}" for i in range(4)])

    # 4. ジャンルペアの嗜好 (171D)
    genre_pair_names = load_genre_pair_names()

    # 最終的なリスト
    all_names = stat_names + aspect_names + ad_names + genre_pair_names

    # 最終的な次元数チェック
    if len(all_names) != USER_FEAT_DIM_FINAL:
        print(f"[ERROR] ユーザー特徴量の合計次元数が{USER_FEAT_DIM_FINAL}Dと一致しません ({len(all_names)}D)。")

    return all_names, 1, 2  # 名前リスト, Mean Index, Std Index


# --- 映画特徴量名生成 (636D) (前回提出コードから変更なし) ---

def get_movie_feature_names() -> List[str]:
    """636Dの映画特徴量の項目名を構築する (CSVヘッダーに基づき修正)"""

    # 0. 元のCSVヘッダー (ユーザー提供情報から取得)
    FULL_CSV_HEADERS = [
        'movie_id', 'movie_title', 'genres', 'release_date', 'production_countries', 'runtime',
        'original_language', 'spoken_languages', 'directors', 'actors', 'keywords', 'A01', 'A02', 'A03',
        'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A17', 'A18', 'A21',
        'A22', 'A25', 'A26', 'A27', 'A28', 'A36', 'A55', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C10',
        'C11', 'C13', 'C16', 'C17', 'C21', 'C22', 'C23', 'C25', 'C26', 'C33', 'C47', 'C53', 'C54', 'D01', 'D02',
        'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09', 'D10', 'D11', 'D12', 'D14', 'D15', 'D42', 'D44', 'D45',
        'D46', 'D47', 'D48', 'E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E07', 'E08', 'E09', 'E10', 'E11', 'E12',
        'E13', 'E14', 'E15', 'E16', 'E17', 'E18', 'E19', 'E20', 'E21', 'E22', 'E26', 'E27', 'E30', 'E35', 'E36',
        'E37', 'E47', 'E48', 'E49', 'E50', 'E53', 'J01', 'J02', 'J03', 'J04', 'J05', 'J06', 'J07', 'J08', 'J09',
        'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20', 'J21', 'J22', 'J23', 'J24',
        'J25', 'J26', 'J27', 'J28', 'J29', 'J30', 'J31', 'J32', 'J33', 'J34', 'J35', 'J36', 'J37', 'J38', 'J39',
        'J40', 'J41', 'J42', 'J43', 'J45', 'J46', 'J47', 'J48', 'J49', 'J50', 'J51', 'J52', 'J53', 'J54', 'J55',
        'P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15',
        'P16', 'P17', 'P18', 'P20', 'P21', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30', 'P31', 'P32',
        'P33', 'P35', 'P36', 'P37', 'P38', 'P39', 'P45', 'P47', 'P50', 'S01', 'S02', 'S03', 'S04', 'S05', 'S06',
        'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14', 'S16', 'S17', 'S18', 'S21', 'S25', 'S26', 'S27',
        'S28', 'S29', 'S30', 'S33', 'S34', 'S35', 'S36', 'S37', 'S40', 'S42', 'S43', 'S44', 'S48', 'S52', 'S53',
        'T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09', 'T10', 'T11', 'T13', 'T14', 'T16', 'T17',
        'T20', 'T27', 'T30', 'T32', 'T35', 'V01', 'V02', 'V03', 'V04', 'V05', 'V06', 'V07', 'V10', 'V11', 'V12',
        'V14', 'V16', 'V42', 'V47', 'V52', 'rating', 'num_raters', 'num_reviews'
    ]

    # 1. 統計的特徴量 (8D)
    stat_names = [
        "movie_review_count_norm", "movie_mean_rating_norm", "movie_std_rating_norm",
        "movie_popularity_score", "movie_release_year_norm", "movie_runtime_norm",
        "movie_budget_norm", "movie_revenue_norm"
    ]

    # 2. 属性特徴量 (263D)
    EXCLUDE_COLS = [
        'movie_id', 'movie_title', 'genres', 'release_date', 'production_countries',
        'runtime', 'original_language', 'spoken_languages', 'directors', 'actors',
        'keywords', 'rating', 'num_raters', 'num_reviews'
    ]
    attribute_cols = [c for c in FULL_CSV_HEADERS if c not in EXCLUDE_COLS]

    # 3. キーワード特徴量 (365D)
    kw_names = []
    # FastText Mean Vector (300D)
    kw_names.extend([f"kw_fasttext_mean_{i:03d}" for i in range(1, 301)])
    # Keyword Count (1D)
    kw_names.append("kw_count")
    # SBERT + PCA (64D)
    kw_names.extend([f"kw_sbert_pca_{i:02d}" for i in range(1, 65)])

    # 最終的な636Dの結合順序: STATS (8D) + ATTRIBUTE (263D) + KEYWORD (365D)
    return stat_names + attribute_cols + kw_names


# --- 分析実行 ---
def analyze_hetero_graph_data(graph_file_path: Path):
    print("=" * 60)
    print("HeteroData Graph Analysis (221D / 636D)")
    print("=" * 60)

    try:
        data = torch.load(graph_file_path, weights_only=False)
        print(f"[SUCCESS] Graph loaded from: {graph_file_path.name}")
    except Exception as e:
        print(f"[ERROR] An error occurred during file loading. Details: {e}")
        return

    # 項目名の取得
    USER_FEATURE_NAMES, user_mean_idx, user_std_idx = get_user_feature_names()
    MOVIE_FEATURE_NAMES = get_movie_feature_names()

    # --- 1. User Node Features ---
    print("\n--- 1. User Node Features ---")
    user_x = data['user'].x
    user_feat_dim = user_x.size(1)

    print(f"  Tensor Shape: {user_x.shape} (N_Users x D_User)")
    print(f"  Data Type: {user_x.dtype}")
    print(f"[CHECK] Expected D_User: {USER_FEAT_DIM_FINAL}D, Actual: {user_feat_dim}D")

    if user_feat_dim != USER_FEAT_DIM_FINAL:
        print(
            f"[CRITICAL] ユーザー特徴量の次元が想定 ({USER_FEAT_DIM_FINAL}D) と異なります。インデックスに注意してください。")

    print("\n  Sample (First User Vector Breakdown - Top 10):")

    # 項目名と実際の次元が異なる場合のチェック
    names_to_use = USER_FEATURE_NAMES[:user_feat_dim]

    print("\n  Sample (Random 5 User Vector Breakdown - Core 6D):")

    # ランダムに5人のユーザーインデックスを選択
    num_users = user_x.size(0)
    rand_indices = torch.randint(0, num_users, (5,)).tolist()

    names_to_use = USER_FEATURE_NAMES[:user_feat_dim]

    # mean/stdのインデックスをチェック
    for i in range(min(user_feat_dim, 10)):
        value = user_x[0, i].item()

        prefix = "    "
        if i == user_mean_idx:
            prefix = "    [MEAN] "
        elif i == user_std_idx:
            prefix = "    [STD]  "
        elif i < 6:
            prefix = "    [CORE] "

        print(f"{prefix}{names_to_use[i]:<30} (Idx {i:02d}): {value:.4f}")



    # --- 2. Movie Node Features ---
    print("\n--- 2. Movie Node Features ---")
    movie_x = data['movie'].x
    movie_feat_dim = movie_x.size(1)

    print(f"  Tensor Shape: {movie_x.shape} (N_Movies x D_Movie)")
    print(f"  Data Type: {movie_x.dtype}")
    print(f"[CHECK] Expected D_Movie: {MOVIE_FEAT_DIM_FINAL}D, Actual: {movie_feat_dim}D")

    # --- 3. Edge Features (Review) ---
    print("\n--- 3. Edge Features (User -> Review -> Movie) ---")
    edge_attr = data['user', 'review', 'movie'].edge_attr
    y_target = data['user', 'review', 'movie'].y

    print(f"  Edge Count (E): {edge_attr.size(0)}")
    print(f"  Edge Attr Shape: {edge_attr.shape} (E_Edges x {EDGE_FEAT_DIM_FINAL}D)")
    print(f"  Target (y) Shape: {y_target.shape} (E_Edges x 1D)")

    # エッジサンプルの詳細
    print("\n--- 4. Edge Sample Breakdown (First Review - Last 3D) ---")

    edge_index = data['user', 'review', 'movie'].edge_index
    print(f"  - Connecting User Index: {edge_index[0, 0].item()}")
    print(f"  - Connecting Movie Index: {edge_index[1, 0].item()}")

    sample_edge_attr = edge_attr[0]

    # rating_user_norm (Index 36) と rating_raw (Index 37) を表示
    if edge_attr.size(1) == EDGE_FEAT_DIM_FINAL:
        print("\n  - Edge Attribute (Crucial Indices):")
        print(f"    [🎯] {ASPECT_COLUMNS[36]:<30} (Idx 36): {sample_edge_attr[36].item():.4f}")
        print(f"    [🎯] {ASPECT_COLUMNS[37]:<30} (Idx 37): {sample_edge_attr[37].item():.4f}")

    print("\n  - Target Label (Y):")
    print(f"    [🎯] Target (rating_user_norm): {y_target[0].item():.4f}")

    # 最終的な結論
    print("\n" + "=" * 60)
    print(f"✅ ユーザー特徴量 (221D) の平均/標準偏差のインデックスは:")
    print(f"   - ユーザー平均評価 (μ) インデックス: {user_mean_idx}")
    print(f"   - 評価標準偏差 (σ) インデックス: {user_std_idx}")
    print("   evaluate_afm.py/train_afm.pyの定数 (USER_MEAN_ABS_INDEX, USER_STD_ABS_INDEX) を")
    print("   それぞれ 1 と 2 に修正する必要があります。")
    print("=" * 60)


if __name__ == "__main__":
    analyze_hetero_graph_data(GRAPH_FILE)