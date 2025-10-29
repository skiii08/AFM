# src/features/inspect_user_info.py (修正版)

import json
from pathlib import Path

# === パス設定 ===
PROJECT_ROOT = Path("/Users/watanabesaki/PycharmProjects/AFM")  # 適切なプロジェクトルートに修正
DATA_PROCESSED = PROJECT_ROOT / "data/processed"
FEATURES_DIR = DATA_PROCESSED / "features"

# ユーザー情報JSONファイルのパス
USER_FEATURE_INFO_JSON = FEATURES_DIR / "user_feature_info.json"


def inspect_problem_user():
    print("=" * 60)
    print("🔎 Inspecting Problematic User (Focusing on ID: 0)")
    print("=" * 60)

    if not USER_FEATURE_INFO_JSON.exists():
        print(f"❌ エラー: 想定されるファイルが見つかりません: {USER_FEATURE_INFO_JSON}")
        return

    with open(USER_FEATURE_INFO_JSON, 'r') as f:
        user_info = json.load(f)

    # 03_build_user_features.py の user_ids リストの最初の要素（User ID: 0 に相当するユーザー）のデータを特定
    # user_infoは user_id(文字列) -> 辞書 の構造であると仮定し、最初の数値キーを持つユーザーを探す

    problem_user_key = None
    # JSONのキーをイテレートし、最初の数字キー（実際のユーザーID）を見つける
    for key in user_info.keys():
        if key.isdigit():
            problem_user_key = key
            break

    if problem_user_key is None:
        print("❌ 致命的エラー: JSONファイルにユーザーIDキー ('1', '2', ...) が見つかりません。ファイル構造が不正です。")
        return

    print(f"✅ JSONの最初のユーザーキー: {problem_user_key} をチェックします。")

    user_data = user_info[problem_user_key]

    print("\n--- User Data (Core Stats) ---")
    print(f"  User ID (Key): {problem_user_key}")
    print(f"  user_name: {user_data.get('user_name', 'N/A')}")
    print(f"  review_count_norm: {user_data.get('review_count_norm')}")
    print(f"  mean_rating: {user_data.get('mean_rating')}")
    print(f"  std_rating: {user_data.get('std_rating')}")
    print(f"  activity_score: {user_data.get('activity_score')}")

    # 0.0 問題が発生したキーをハイライト
    if user_data.get('mean_rating') == 0.0 or (
            isinstance(user_data.get('mean_rating'), float) and np.isnan(user_data.get('mean_rating'))):
        print("\n🚨🚨🚨 mean_rating の値が不正です。これがエラーの原因です。🚨🚨🚨")


if __name__ == "__main__":
    # numpyのNaN比較のための一時的な設定
    try:
        inspect_problem_user()
    except Exception as e:
        print(f"\n予期せぬエラー: {e}")