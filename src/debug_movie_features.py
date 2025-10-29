# src/debug/debug_movie_features.py (最終修正版 - PyTorch 2.8.0対応)

import torch
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict

# PyTorch Geometricのカスタムクラスをロードするために必要
from torch_geometric.data import HeteroData

# === PyTorchの安全設定 (最も重要) ===
# PyTorch 2.6以降のデフォルト設定変更に対応するため、カスタムクラスを安全なグローバルとして登録します。
# これにより、HeteroDataオブジェクトをweights_only=Trueでもロードできます。
# ただし、今回は明示的にweights_only=Falseを使うため、予防的な措置となります。
try:
    # torch_geometric.data.hetero_data.HeteroData を許可リストに追加
    torch.serialization.add_safe_globals([HeteroData])
except AttributeError:
    # 互換性のため、add_safe_globalsがない場合はスキップ
    pass

# === パス設定 (FINAL build_movie_features.py と同じ設定に合わせる) ===
# 実行環境に合わせて適宜修正してください
PROJECT_ROOT = "/Users/watanabesaki/PycharmProjects/AFM"
DATA_PROCESSED = Path(PROJECT_ROOT) / "data/processed"
FEATURES_DIR = DATA_PROCESSED / "features"

MOVIE_FEATURES_PT = DATA_PROCESSED / "movie_features.pt"
MOVIE_FEATURE_INFO_JSON = FEATURES_DIR / "movie_feature_info.json"
TRAIN_GRAPH_PT = DATA_PROCESSED / "hetero_graph_train.pt"


# --- 補助関数 ---
def load_and_validate_movie_features():
    """movie_features.ptとメタデータをロードし、次元と統計情報を検証"""
    print("=" * 60)
    print("1. MOVIE FEATURES (.pt) 検証")
    print("=" * 60)

    if not MOVIE_FEATURES_PT.exists():
        print(f"❌ エラー: 特徴量ファイルが見つかりません: {MOVIE_FEATURES_PT}")
        print("    'FINAL build_movie_features.py' が正常に実行されたか確認してください。")
        return None, None

    try:
        # UnpicklingError対策: weights_only=Falseを明示的に指定
        features = torch.load(MOVIE_FEATURES_PT, weights_only=False)
        feat_np = features.numpy()  # 最初にNumpy変換
    except Exception as e:
        print(f"❌ エラー: 特徴量ファイルのロードに失敗しました。詳細: {e}")
        return None, None

    print(f"✅ ロード成功: {MOVIE_FEATURES_PT.name}")
    print(f"    Shape: {features.shape}")

    # 特徴量情報ファイル（JSON）のロード
    if not MOVIE_FEATURE_INFO_JSON.exists():
        print(f"❌ エラー: 情報ファイルが見つかりません: {MOVIE_FEATURE_INFO_JSON}")
        return features, None

    try:
        with open(MOVIE_FEATURE_INFO_JSON, 'r') as f:
            info = json.load(f)
    except Exception as e:
        print(f"❌ エラー: 情報ファイルのロードに失敗しました。詳細: {e}")
        return features, None

    # 全体統計の確認
    print("\n--- 全体統計 ---")
    print(f"  Max/Min: {feat_np.max():.4f} / {feat_np.min():.4f}")
    print(f"  Mean/Std: {feat_np.mean():.4f} / {feat_np.std():.4f}")

    # キーワードブロックの検証 (元のエラー回避)
    offsets = info.get('offsets', {})
    if 'keyword_hybrid' in offsets:
        kw_offset = offsets['keyword_hybrid']
        kw_feats = features[:, kw_offset['start']:kw_offset['end']]
        kw_np = kw_feats.numpy()
        zero_rows = np.all(np.isclose(kw_np, 0.0), axis=1)
        zero_ratio = zero_rows.sum() / len(kw_np)

        print("\n--- キーワード特徴量 (Hybrid) 検証 ---")
        print(f"  キーワード次元: {kw_offset['dim']}D (期待値: 365D)")

        if zero_ratio > 0.5:
            print(f"❌ 警告: 全映画の {zero_ratio * 100:.1f}% がゼロベクトルです。")
        else:
            print(f"✅ キーワードの特徴量は適切に分散しています。ゼロベクトル率: {zero_ratio * 100:.1f}%")
    else:
        # キーワード特徴量がない場合の警告を出力
        print("\n❌ 警告: 特徴量情報ファイルに 'keyword_hybrid' のオフセット情報がありません。")
        print("    '04_build_movie_features.py' でキーワード特徴量が生成/保存されていない可能性があります。")
        print("    現在の特徴量は 271D で、キーワード特徴量 (約365D) が欠落しています。")

    return features, info


def validate_graph(movie_features: torch.Tensor, movie_info: Dict):
    """グラフファイル内の映画ノード特徴の値を検証"""
    print("\n" + "=" * 60)
    print("2. グラフファイル (.pt) 検証")
    print("=" * 60)

    if movie_features is None:
        print("❌ エラー: movie_features がロードされていないため、グラフ検証をスキップします。")
        return

    if not TRAIN_GRAPH_PT.exists():
        print(f"❌ エラー: グラフファイルが見つかりません: {TRAIN_GRAPH_PT}")
        return

    try:
        # UnpicklingError対策: weights_only=Falseを明示的に指定
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            graph = torch.load(TRAIN_GRAPH_PT, weights_only=False)  # <--- ここを修正
    except Exception as e:
        print(f"❌ エラー: グラフファイルのロードに失敗しました。")
        print(f"    原因: {type(e).__name__}: {e}")  # <--- エラーの型とメッセージを出力
        return

    print(f"✅ ロード成功: {TRAIN_GRAPH_PT.name}")

    if 'movie' in graph.node_types and hasattr(graph['movie'], 'x'):
        graph_movie_feat = graph['movie'].x
        expected_dim = movie_features.shape[1]

        print(f"\n--- グラフ組み込み特徴量詳細 ---")
        print(f"  グラフ内の映画ノード数: {graph_movie_feat.shape[0]}")
        print(f"  グラフ内の映画ノード特徴のShape: {graph_movie_feat.shape}")

        # 映画特徴量の詳細な内訳を報告
        feat_dim = movie_features.shape[1]
        stat_dim = 8  # 統計特徴量（runtime, ratingなど）
        attr_dim = feat_dim - stat_dim

        print(f"  総次元数: {feat_dim}D")
        print(f"    - 統計特徴量 (Stats): {stat_dim}D")
        print(f"    - 属性特徴量 (Attributes): {attr_dim}D")

        # 統計量の再チェック
        graph_feat_np = graph_movie_feat.numpy()
        print("\n--- グラフ組み込み特徴量の全体統計 ---")
        print(f"  Max/Min: {graph_feat_np.max():.4f} / {graph_feat_np.min():.4f}")
        print(f"  Mean/Std: {graph_feat_np.mean():.4f} / {graph_feat_np.std():.4f}")

        if graph_movie_feat.shape[1] == expected_dim:
            # 組み込まれた特徴量と生成した特徴量を比較
            diff = torch.abs(graph_movie_feat[0] - movie_features[0]).sum().item()

            if diff < 1e-4:
                print("\n✅ 検証結果: 特徴量の値は**最初の行で正確に一致**しています。")
                print("    特徴量データ自体は正常にグラフに組み込まれています。")
            else:
                print(f"\n❌ 致命的なエラー: 生成特徴量とグラフの特徴量の値が不一致。差: {diff:.6f}")
                print("    '05_build_graph.py' のマッピングロジックに問題がある可能性があります。")
        else:
            print(
                f"\n❌ 致命的なエラー: グラフの特徴量次元 ({graph_movie_feat.shape[1]}D) が期待値 ({expected_dim}D) と異なります。")
    else:
        print("❌ 致命的なエラー: グラフ内に映画ノードまたは特徴量 (x) が見つかりません。")


# --- メイン処理 ---
if __name__ == '__main__':
    # 1. movie_features.pt の検証
    features, info = load_and_validate_movie_features()

    # 2. グラフの検証
    if features is not None and info is not None:
        # warnings.warn() を抑制するため、urllib3の警告を一時的に無視
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*NotOpenSSLWarning.*")
            validate_graph(features, info)

    print("\nプロセスは終了コード 0 で終了しました")