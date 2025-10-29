import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm  # 処理進捗表示用
from typing import List

# === パス設定 (ユーザーの環境に合わせて修正してください) ===
# 実行ファイル(分析用) -> src(1) -> AFM(2)
PROJECT_ROOT = Path("/Users/watanabesaki/PycharmProjects/AFM")  # 💥 実行環境に応じて修正
DATA_RAW = PROJECT_ROOT / "data/raw"

# 入力ファイルパス
MOVIES_CSV = DATA_RAW / "movies_metadata.csv"

# 属性カラムを特定するための除外リスト (04_build_movie_features.pyより)
EXCLUDE_COLS = [
    'movie_id', 'movie_title', 'genres', 'release_date', 'production_countries',
    'runtime', 'original_language', 'spoken_languages', 'directors', 'actors',
    'keywords', 'rating', 'num_raters', 'num_reviews'
]


def get_attribute_columns(df: pd.DataFrame) -> List[str]:
    """属性タグカラム (A01-V52) のリストを取得する"""
    # ヘッダーから除外リストにないカラムを抽出
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def analyze_attribute_distribution():
    print("=" * 70)
    print("Movie Attribute Distribution Analysis (A01 - V52)")
    print("=" * 70)

    if not MOVIES_CSV.exists():
        print(f"[ERROR] 映画メタデータファイルが見つかりません: {MOVIES_CSV}")
        return

    try:
        # メタデータを読み込み、NaN（欠損値）は0として扱う（タグデータは欠損＝0と想定）
        movies_df = pd.read_csv(MOVIES_CSV, low_memory=False).fillna(0)
    except Exception as e:
        print(f"[ERROR] ファイル読み込み中にエラーが発生しました: {e}")
        return

    attribute_cols = get_attribute_columns(movies_df)
    print(f"✅ {len(attribute_cols)}個の属性カラム (A01 - V52) を特定しました。")

    results = []

    # 統計情報を計算
    print("\n[Analysis] Calculating statistics...")
    for col in tqdm(attribute_cols, desc="Processing Columns"):
        series = movies_df[col]
        # 数値型に変換
        try:
            series = pd.to_numeric(series, errors='coerce').fillna(0)
        except:
            continue

        if series.max() <= 0:
            results.append({"Column": col, "Max Z-Score (Approx)": 0.0, "Status": "All Zeroes"})
            continue

        # 統計量の計算
        max_val = series.max()
        std_val = series.std()
        mean_val = series.mean()
        p95 = series.quantile(0.95)
        p99 = series.quantile(0.99)

        # Z-Scoreの最大値を計算: (最大値 - 平均) / 標準偏差
        # これが、Z-Score正規化後の最大値の目安となります。
        max_z_score_approx = (max_val - mean_val) / std_val if std_val > 0 else 0.0

        results.append({
            "Column": col,
            "Max": max_val,
            "Median": series.median(),
            "95th %": p95,
            "99th %": p99,
            "Max Z-Score (Approx)": max_z_score_approx,
            "Status": "OK"
        })

    # 結果をDataFrameに変換し、Max Z-Scoreが大きい順にソート
    results_df = pd.DataFrame([r for r in results if r.get('Status') == 'OK'])
    results_df = results_df.sort_values(by="Max Z-Score (Approx)", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("【結果1: Max Z-Scoreに基づく偏りのトップ10】")
    print("  ※この値が現在の16.8671の原因です。5.0を超えるとLog変換を検討すべきです。")
    print("-" * 70)
    print(results_df.head(10).to_string(float_format="%.4f"))

    # E48 の情報を探して表示
    e48_info = results_df[results_df['Column'] == 'E48']
    if not e48_info.empty:
        print("\n[E48] E48 カラムの詳細:")
        print(e48_info[['Max', '99th %', 'Max Z-Score (Approx)']].to_string(float_format="%.4f"))

    print("\n" + "=" * 70)
    print("【次のアクションの検討】")
    print("-" * 70)

    # E48のMax Z-Scoreを取得
    e48_max_z = e48_info['Max Z-Score (Approx)'].iloc[0] if not e48_info.empty else 0

    if e48_max_z > 10.0 or results_df['Max Z-Score (Approx)'].head(10).mean() > 5.0:
        print(f"📈 強い偏りを確認 (E48 Max Z-Score: {e48_max_z:.2f})")
        print(
            "  → **Z-Score正規化の前にLog(1+x)変換を適用**することで、この極端な値を抑制し、モデルの学習を安定させることができます。")
    else:
        print("✅ 偏りはZ-Score化で許容範囲内です。")
        print("  → Z-Score化のみで十分な可能性がありますが、もしLog変換を試したい場合は次のステップに進んでください。")

    print("=" * 70)


if __name__ == "__main__":
    try:
        # tqdmがインストールされていない場合はダミー関数を使用
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, *args, **kwargs):
            return iterable

    analyze_attribute_distribution()