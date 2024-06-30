import json
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from utility import load_config, summarize_object_columns


def store_column_roles_json(df, json_filepath, target_columns, key_columns):
    """
    DataFrameのカラムの詳細をJSON形式で保存する関数。

    この関数は、指定されたDataFrameの各カラムについての詳細情報を
    JSON形式で保存します。"usage"フィールドの値は以下の
    ルールに従って設定されます:

    - "target": 引数 target_columns で指定したカラム
    - "key": 引数 key_columns で指定したカラム
    - null: データ型が object のカラム
    - "feature": それ以外のカラム

    Parameters:
    df (pandas.DataFrame): カラムの詳細を保存する対象のDataFrame。
    json_filepath (str): 保存するJSONファイルのパス。
    target_columns (list): "target"としてマークするカラムのリスト。
    key_columns (list): "key"としてマークするカラムのリスト。

    Returns:
    None
    """
    columns_details = {
        col: {
            "dtype": str(df[col].dtype),
            "nunique": df[col].nunique(),
            "usage": "target"
            if col in target_columns
            else "key"
            if col in key_columns
            else (None if df[col].dtype == "object" else "feature"),
        }
        for col in df.columns
    }
    with open(json_filepath, "w", encoding="utf-8") as file:
        json.dump(columns_details, file, ensure_ascii=False, indent=4)
    print("Saved columns list to json:", os.path.basename(json_filepath))


# 環境変数の読み込み
project_path = "../../"
env_file = os.getenv("ENV_FILE", os.path.join(project_path, ".env"))
load_dotenv(env_file)
file_directory = os.getenv("DF_DIR")
since_year = 2015

# 設定ファイルの読込み
config = load_config(os.path.join(project_path, "config.yaml"))
since_year = int(str(config["initial_dataset_year"]))
pred_date = int(str(config["prediction_date"]))

# データの読込み
read_file_path = os.path.join(str(file_directory), "ALL_DATA.feather")
df = pd.read_feather(read_file_path)

# データの日付による絞り込み
df["IN_年月日"] = pd.to_numeric(df["IN_年月日"], errors="coerce").astype(int)
df = df[df["IN_年月日"] >= since_year * 10000].reset_index(drop=True)
df["IN_年月日"] = df["IN_年月日"].astype(str)

df_summary = summarize_object_columns(df)

# 保存先JSONファイルパス
save_json_path = os.path.join(str(project_path), "column_info.json")

# カラムを説明変数と目的変数に振り分けるためのJSONファイルを作成する
target_columns = [
    "OUT_馬成績_着順",
    "OUT_馬成績_確定単勝オッズ",
    "OUT_確定複勝オッズ下",
]
key_columns = [
    "IN_レースキー",
    "IN_レースキー_場コード",
    "IN_レースキー_回",
    "IN_レースキー_日",
    "IN_レースキー_R",
    "_merge",
]
store_column_roles_json(df, save_json_path, target_columns, key_columns)

# 読込先JSONファイルパス
read_json_path = os.path.join(str(project_path), "column_info_fixed.json")

# JSONファイルを読み込む
with open(read_json_path, "r", encoding="utf-8") as file:
    columns_info = json.load(file)
print("Loaded columns list from json:", os.path.basename(read_json_path))
# JSONの情報を元にカラムを振り分ける
target_columns = [
    col for col, details in columns_info.items() if details["usage"] == "target"
]
feature_columns = [
    col
    for col, details in columns_info.items()
    if details["usage"] in ("feature", "key")
]
# 予測データを分離
df["IN_年月日"] = pd.to_numeric(df["IN_年月日"], errors="coerce").astype(int)
df_train = df[(df["_merge"] == 0) & (df["IN_年月日"] < pred_date)].reset_index(
    drop=True
)
df_test = df[df["IN_年月日"] == pred_date]
# 目的変数データと説明変数データを抽出
df_target = df_train[target_columns]
df_features = df_train[feature_columns]
df_test_features = df_test[feature_columns]

# 1着の馬を1とする
df_target = df_target.copy()
df_target["単勝"] = np.where(df_target["OUT_馬成績_着順"] == 1, 1, 0)

# 3着以内に入る馬を1とする
df_target = df_target.copy()
df_target["複勝"] = np.where(df_target["OUT_馬成績_着順"] <= 3, 1, 0)

# 1着かつ単勝オッズ5倍以上の馬を1とする
df_target["単勝穴馬"] = (
    (df_target["OUT_馬成績_着順"] == 1) & (df_target["OUT_馬成績_確定単勝オッズ"] >= 5)
).astype(int)

# 3着以内かつ複勝オッズ3倍以上の馬を1とする
df_target["複勝穴馬"] = (
    (df_target["OUT_馬成績_着順"] <= 3) & (df_target["OUT_確定複勝オッズ下"] >= 3)
).astype(int)

# 着順カテゴリを作成する
conditions = [
    df_target["OUT_馬成績_着順"].isna(),
    df_target["OUT_馬成績_着順"] == 1,
    df_target["OUT_馬成績_着順"] == 2,
    df_target["OUT_馬成績_着順"] == 3,
    (df_target["OUT_馬成績_着順"] >= 4) & (df_target["OUT_馬成績_着順"] <= 10),
    df_target["OUT_馬成績_着順"] >= 11,
]
values = [0, 30, 25, 20, 5, 1]
df_target["着順カテゴリ"] = np.select(conditions, values)

# ファイル保存
target_file_path = os.path.join(str(file_directory), "DF_TARGET.feather")
features_file_path = os.path.join(str(file_directory), "DF_FEATURES.feather")
test_features_file_path = os.path.join(str(file_directory), "DF_TEST_FEATURES.feather")
df_target.to_feather(target_file_path)
df_features.to_feather(features_file_path)
df_test_features.to_feather(test_features_file_path)
