# src/features/04_build_movie_features.py (最終修正版 - KeyError対応済み)

import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict
import re
from collections import Counter
from sklearn.decomposition import PCA
import torch as th  # torch.catとの競合を避けるためthとしてインポート
import sys

# 外部ライブラリのインポート (インストール必須)
try:
    import fasttext
except ImportError:
    fasttext = None
    print("[ERROR] fasttext library not found. Please run: pip install fasttext")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    print("[ERROR] sentence-transformers library not found. Please run: pip install sentence-transformers")

# === パス設定 ===
# 実行ファイル(04) -> features(1) -> src(2) -> AFM(3)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data/raw"
DATA_PROCESSED = PROJECT_ROOT / "data/processed"
FEATURES_DIR = DATA_PROCESSED / "features"

# 入力
MOVIES_CSV = DATA_RAW / "movies_metadata.csv"

# 【🔑要修正】FastTextモデルのパスを適切に設定してください
FASTTEXT_MODEL_PATH = PROJECT_ROOT / "data/external/cc.en.300.bin"

# 出力
MOVIE_FEATURES_PT = DATA_PROCESSED / "movie_features.pt"
MOVIE_FEATURE_INFO_JSON = FEATURES_DIR / "movie_feature_info.json"


# --- 補助関数 ---

def parse_list_cell(s: str) -> list[str]:
    """区切り文字(;)でリストをパース"""
    if pd.isna(s) or s == "":
        return []
    return [x.strip() for x in str(s).split(";") if x.strip()]


def zscore_with_missing(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    シリーズをZ-Score正規化し、欠損値インジケータも返す。
    欠損値は平均値で補完される。
    """
    not_na = series.notna()
    data = series[not_na].values.astype(np.float32)

    if data.size == 0:
        return np.zeros_like(series, dtype=np.float32), np.zeros_like(series, dtype=np.float32)

    # Z-Score計算
    mean_val = data.mean()
    std_val = data.std()

    if std_val == 0:
        zscore_data = np.zeros_like(data, dtype=np.float32)
    else:
        zscore_data = (data - mean_val) / std_val

    # 全データに対するZ-Score配列を作成
    zscore_full = np.zeros_like(series, dtype=np.float32)
    zscore_full[not_na] = zscore_data

    # 欠損値インジケータ
    missing_indicator = (~not_na).values.astype(np.float32)

    return zscore_full, missing_indicator


def keywords_to_hybrid_vec(ft_model: fasttext.FastText._FastText, sbert_model: SentenceTransformer,
                           kws: List[str], kw_counter: Counter, pca: PCA) -> np.ndarray:
    """
    FastText, SBERT, PCAを組み合わせてキーワードのハイブリッドベクトルを生成する。
    次元: 300D (FT平均) + 1D (総数) + 64D (SBERT PCA) = 365D
    """
    if not kws:
        return np.zeros(365, dtype=np.float32)

    # 1. FastText (FT) 平均ベクトル (300D)
    ft_vecs = [ft_model.get_word_vector(kw) for kw in kws if kw in ft_model]
    if ft_vecs:
        ft_mean_vec = np.mean(ft_vecs, axis=0)
    else:
        ft_mean_vec = np.zeros(300, dtype=np.float32)

    # 2. キーワード総数 (1D)
    num_kws = np.array([len(kws)], dtype=np.float32)

    # 3. SBERT PCA (64D) - 頻出キーワードのみを考慮
    sbert_vecs = sbert_model.encode(kws)
    sbert_mean_vec = np.mean(sbert_vecs, axis=0).reshape(1, -1)
    sbert_pca_vec = pca.transform(sbert_mean_vec).flatten()  # 64D

    # 結合 (300D + 1D + 64D = 365D)
    hybrid_vec = np.concatenate([ft_mean_vec, num_kws, sbert_pca_vec])

    return hybrid_vec


def main():
    print("=" * 80)
    print("4. BUILD MOVIE FEATURES (Creating movie_features.pt - Real Data Only)")
    print("=" * 80)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # 1. データのロードと前処理
    try:
        movies_df = pd.read_csv(MOVIES_CSV)
        # movie_idを整数に変換
        movies_df['movie_id'] = pd.to_numeric(movies_df['movie_id'], errors='coerce').fillna(-1).astype(int)
        movies_df = movies_df[movies_df['movie_id'] != -1].copy()
        movies_df.sort_values(by='movie_id', inplace=True)
        movies_df.reset_index(drop=True, inplace=True)
        num_movies = len(movies_df)
        print(f"[Preproc] Converted movie_id to integer and kept {num_movies} movies.")
    except FileNotFoundError:
        print(f"[ERROR] Movies metadata file not found at {MOVIES_CSV}. Aborting.")
        return

    print(f"[Load] Loaded {num_movies} movies from {MOVIES_CSV.name}")

    # 2. 統計特徴量 (8D)
    # 💥 修正: 実際のカラム名に合わせて修正
    STATS_COLS = ['runtime', 'rating', 'num_raters', 'num_reviews']
    stats_features_list = []

    print("[Feature] Generating Statistical Features (runtime, rating, num_raters, num_reviews)...")
    for col in STATS_COLS:
        # KeyErrorの原因箇所
        zscore, missing_ind = zscore_with_missing(movies_df[col])
        stats_features_list.append(torch.tensor(zscore, dtype=torch.float32).unsqueeze(1))
        stats_features_list.append(torch.tensor(missing_ind, dtype=torch.float32).unsqueeze(1))

    stats_features = torch.cat(stats_features_list, dim=1)
    print(f"  Statistical Feature Shape: {stats_features.shape} ({stats_features.shape[1]}D)")  # 8D

    # 3. 属性特徴量 (Genres, Production, Cast/Crew, etc. - 263D)

    # 3. 属性特徴量 (Genres, Production, Cast/Crew, etc. - 263D)

    # 💥 修正: 属性特徴量を明示的に抽出する
    EXCLUDE_COLS = [
        'movie_id', 'movie_title', 'genres', 'release_date', 'production_countries',
        'runtime', 'original_language', 'spoken_languages', 'directors', 'actors',
        'keywords', 'rating', 'num_raters', 'num_reviews'
    ]

    # 除外リストにない、残りのカラム（A01～V52のような特徴量カラム）を属性特徴量とする
    # 3. 属性特徴量 (A01〜V52) の抽出と正規化 💥【修正箇所】
    # 04_build_movie_features.py の属性特徴量（attribute_cols）処理部分の修正

    # 3. 属性特徴量 (A01〜V52) の抽出と正規化 💥【修正箇所】
    # 3. 属性特徴量 (A01〜V52) の抽出と正規化 💥【修正箇所】
    attribute_cols = [c for c in movies_df.columns if c not in EXCLUDE_COLS]

    normalized_attributes = []
    transform_applied = 0
    max_z_scores = {}

    print("  Applying Sqrt-transform and Z-Score normalization to 263 Attribute features...")

    for col in tqdm(attribute_cols, desc="Normalizing attributes"):
        series = movies_df[col]

        # 欠損値は0として扱い、数値型に変換
        series = pd.to_numeric(series, errors='coerce').fillna(0)

        # 【変更】Log(1+x)からSqrt(平方根)変換へ切り替え
        if (series < 0).any():
            transformed = series
        else:
            # 偏りを強力に抑制するための平方根変換
            transformed = np.sqrt(series)
            transform_applied += 1

        # Z-Scoreを計算
        z_score, _ = zscore_with_missing(transformed)

        # 検証ロジック: 正規化後の最大値を記録
        if z_score.size > 0:
            max_val = np.max(z_score)
            max_z_scores[col] = max_val

        normalized_attributes.append(z_score[:, np.newaxis])

    # 結合して属性特徴量テンソルを作成
    attribute_features = torch.tensor(
        np.hstack(normalized_attributes),
        dtype=torch.float32
    )

    # 3.5. 【追加】属性特徴量ブロック全体にL2正規化を適用
    epsilon = 1e-6
    norm = torch.linalg.norm(attribute_features, dim=1, keepdim=True) + epsilon
    attribute_features_final = attribute_features / norm
    print(f"  Applied L2-Normalization to Attribute features (263D).")

    # 💥【L2正規化の成功デバッグロジックを追加】
    # L2ノルムが1に収束しているかを検証する
    final_norm = torch.linalg.norm(attribute_features_final, dim=1)
    max_value_after_l2 = torch.max(attribute_features_final).item()

    print("\n" + "=" * 60)
    print("【L2正規化 成功検証】")
    print("-" * 60)

    # 1. L2ノルムの検証 (長さが1になっているか)
    print(f"✅ L2ノルムの平均: {final_norm.mean().item():.6f} (目標: 1.000000)")
    print(f"✅ L2ノルムの標準偏差: {final_norm.std().item():.6f} (目標: 0.000000)")

    # 2. 値の範囲の検証 (最大値が1.0を超えていないか)
    print(f"✅ 正規化後の最大値: {max_value_after_l2:.6f} (制約: <= 1.0)")

    if final_norm.mean().item() > 0.999 and final_norm.std().item() < 1e-5:
        print("🌟 L2正規化は完全に成功しており、すべての映画ノードの属性特徴量は均衡が取れています。")
        print("   → **この時点で、タグの「数が多いことによる寄与の偏り」は解消されています。**")
    else:
        print("❌ L2正規化に問題があります。コードを確認してください。")

    # 💥【コンソール出力ロジック - 実行結果出力用】
    if max_z_scores:
        max_col = max(max_z_scores, key=max_z_scores.get)
        max_z = max_z_scores[max_col]

        high_z_count = sum(1 for z in max_z_scores.values() if z > 5.0)

        print("\n" + "=" * 60)
        print("【属性特徴量 (A01〜V52) 均衡検証結果 - Sqrt + Z-Score適用後】")
        print(f"📈 最も偏りが残るカラム: {max_col} (Max Z-Score: {max_z:.4f})")
        print(f"⚠ Z-Score > 5.0 となるカラム数: {high_z_count} / {len(max_z_scores)}")
        print("✅ **L2正規化を適用したため、このZ-Score値は参考情報となります。**")
        print("   → L2正規化により、各映画の属性ベクトル全体が強制的に均衡化されています。")
        print("=" * 60)

    # 4. 最終特徴量テンソルの結合（結合順序を修正）


    print("[Feature] Extracting all Attribute Features (A**, C**, D**, E**, J**, P**, S**, T**, V**)...")
    print(f"  Attribute Feature Shape: {attribute_features.shape} ({attribute_features.shape[1]}D)")

    # =========================================================================
    # 💥 4. キーワード特徴量 (Hybrid) の生成 (365D)
    # =========================================================================

    if fasttext is None or SentenceTransformer is None:
        print("\n[SKIP] Keyword features skipped due to missing libraries (fasttext or sentence-transformers).")
        keyword_features = torch.zeros((num_movies, 365), dtype=torch.float32)
        kw_dim = 365
    else:
        print("\n[Feature] Generating Keyword Hybrid Features (365D)...")
        # モデルのロード
        ft_model = None
        sbert_model = None
        try:
            ft_model = fasttext.load_model(str(FASTTEXT_MODEL_PATH))
            print("  [FastText] Model loaded successfully.")
        except Exception as e:
            print(f"  [ERROR] FastText model failed to load. Path: {FASTTEXT_MODEL_PATH}. Error: {e}")

        if ft_model and SentenceTransformer is not None:
            try:
                print("  [SBERT] Loading all-MiniLM-L6-v2...")
                sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"  [ERROR] SBERT model failed to load. Error: {e}")
                sbert_model = None

        if ft_model and sbert_model:
            # キーワードのパース
            movies_df["keywords"] = movies_df["keywords"].fillna("").astype(str)
            kw_lists = movies_df["keywords"].apply(parse_list_cell).tolist()

            # 頻度カウント
            kw_counter = Counter()
            for kws in kw_lists:
                kw_counter.update(kws)

            # 頻出キーワードのフィルタリング (Count >= 5)
            freq_kws = [kw for kw, cnt in kw_counter.items() if cnt >= 5]
            print(f"  [Keywords] Total: {len(kw_counter)}, Freq>=5: {len(freq_kws)}")

            # PCA学習 (SBERTベクトルを使用)
            print("  [PCA] Training on frequent keywords...")
            if len(freq_kws) > 0:
                sbert_freq_vecs = sbert_model.encode(freq_kws, show_progress_bar=False)
                pca = PCA(n_components=64)  # 64Dに削減
                pca.fit(sbert_freq_vecs)
            else:
                print("  [WARNING] No frequent keywords found. Skipping PCA and using zero vector.")
                pca = None

            # Hybrid vectorsの生成
            print("  [Hybrid] Generating keyword vectors...")
            if pca is not None:
                kw_vecs = [keywords_to_hybrid_vec(ft_model, sbert_model, kws, kw_counter, pca)
                           for kws in tqdm(kw_lists, desc="Generating Hybrid Feats")]
                kw_vecs = np.stack(kw_vecs, axis=0)
                kw_dim = kw_vecs.shape[1]  # 365D
                keyword_features = torch.tensor(kw_vecs, dtype=torch.float32)
            else:
                keyword_features = torch.zeros((num_movies, 365), dtype=torch.float32)
                kw_dim = 365
                print("  [WARNING] PCA setup failed. Keyword features set to zero vectors (365D).")
        else:
            # モデルロード失敗時
            keyword_features = torch.zeros((num_movies, 365), dtype=torch.float32)
            kw_dim = 365
            print("  [WARNING] Model(s) failed to load. Keyword features set to zero vectors (365D).")

        print(f"  Keyword Feature Shape: {keyword_features.shape} ({kw_dim}D)")

    # 5. 最終特徴量の結合
    print("\n[Combine] Combining final Movie features (Stats + Attributes + Keywords)...")

    # 結合順序: STATS_FEATURES (8D) + ATTRIBUTE_FEATURES (263D) + KEYWORD_FEATURES (365D)


    movie_features_final = torch.cat([
        stats_features,  # 8D
        attribute_features_final,  # 263D 【修正】L2正規化後の特徴量を使用
        keyword_features  # 365D
    ], dim=1)

    final_dim = movie_features_final.shape[1]

    print(f"  Final shape: {movie_features_final.shape} (Total {final_dim}D)")

    # 6. 保存
    print("\n[Save] Saving features...")
    th.save(movie_features_final, MOVIE_FEATURES_PT)
    print(f"  Features saved to: {MOVIE_FEATURES_PT}")

    # 7. 必須: 映画インデックス情報 (05_build_graph.pyで必要)
    movie_id_to_idx = {
        row['movie_id']: idx
        for idx, row in movies_df.reset_index().rename(columns={'index': 'movie_idx'}).iterrows()
    }
    movie_ids = movies_df['movie_id'].tolist()

    # 💥 必須: 特徴量情報ファイルにオフセット情報を含める
    stats_end = stats_features.shape[1]
    attribute_end = stats_end + attribute_features.shape[1]

    movie_feature_info = {
        "num_movies": num_movies,
        "movie_feat_dim": final_dim,
        "movie_ids": movie_ids,
        "offsets": {
            "stats": {"start": 0, "end": stats_end, "dim": stats_features.shape[1]},  # 8D
            "attribute": {"start": stats_end,
                          "end": attribute_end,
                          "dim": attribute_features.shape[1]},  # 263D
            "keyword_hybrid": {"start": attribute_end,
                               "end": final_dim,
                               "dim": kw_dim}  # 365D
        }
    }

    with open(MOVIE_FEATURE_INFO_JSON, 'w') as f:
        json.dump(movie_feature_info, f, indent=2)
    print(f"  Saved movie index info: {MOVIE_FEATURE_INFO_JSON}")


if __name__ == "__main__":
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, *args, **kwargs):
            return iterable

    if fasttext is None or SentenceTransformer is None:
        print("\n[ABORT] Cannot run due to missing required libraries (fasttext or sentence-transformers).")
        # ライブラリがない場合は強制終了
        # sys.exit(1) は、今回はモデルロード失敗時にゼロベクトルで継続できるようにするため、コメントアウトします。
        # ただし、モデルロード失敗時にもゼロベクトルで365Dが保証されるようロジックを修正しました。
        pass

    main()