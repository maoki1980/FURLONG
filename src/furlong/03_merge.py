import os

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from utility import summarize_object_columns


# featherファイルを読み込む関数
def read_df_file(file_directory, file_category):
    file_path = os.path.join(file_directory, f"{file_category}_fixed.feather")
    df = pd.read_feather(file_path)
    return df


# 競走馬データに過去n走までの前走成績データを結合する関数
def merge_previous_race_data(df_horse, df_result, n):
    df = df_horse.copy()

    for i in tqdm(range(1, n + 1), desc="Processing dataframe"):
        horse_key = f"他データリンク用キー_前走{i}競走成績キー"
        result_prefix = f"{i}走前_"
        result_key = result_prefix + "競走成績キー"

        df_res_copy = df_result.copy()
        df_res_copy.columns = [result_prefix + col for col in df_res_copy.columns]

        df = pd.merge(
            df, df_res_copy, left_on=[horse_key], right_on=[result_key], how="left"
        )

    return df


# 環境変数の読み込み
project_path = "../../"
env_file = os.getenv("ENV_FILE", os.path.join(project_path, ".env"))
load_dotenv(env_file)
file_directory = os.getenv("DF_DIR")


df_bac = read_df_file(file_directory, "BAC")
df_hjc = read_df_file(file_directory, "HJC")
df_kab = read_df_file(file_directory, "KAB")
df_kyi = read_df_file(file_directory, "KYI")
df_sed = read_df_file(file_directory, "SED")

df_race = pd.merge(
    df_bac,
    df_kab,
    left_on=[
        "レースキー_場コード",
        "レースキー_年",
        "レースキー_回",
        "レースキー_日",
        "年月日",
    ],
    right_on=[
        "開催キー_場コード",
        "開催キー_年",
        "開催キー_回",
        "開催キー_日",
        "年月日",
    ],
    how="inner",
)
df_race = df_race.drop(
    columns=["開催キー_場コード", "開催キー_年", "開催キー_回", "開催キー_日"], axis=1
)

df_horse = pd.merge(
    df_kyi,
    df_race,
    on=[
        "レースキー_場コード",
        "レースキー_年",
        "レースキー_回",
        "レースキー_日",
        "レースキー_R",
    ],
    how="inner",
)

df_output = pd.merge(
    df_sed,
    df_race,
    on=[
        "レースキー_場コード",
        "レースキー_年",
        "レースキー_回",
        "レースキー_日",
        "レースキー_R",
    ],
    how="inner",
)

df_input = merge_previous_race_data(df_horse, df_output, 3)

# カラム名に接頭辞を追加
prefix = "IN_"
df_input.columns = [prefix + col for col in df_input.columns]
prefix = "OUT_"
df_output.columns = [prefix + col for col in df_output.columns]

# アウトプット (目的変数) から必要な情報を抽出
df_output_subset = df_output[
    [
        "OUT_レースキー_場コード",
        "OUT_レースキー_年",
        "OUT_レースキー_回",
        "OUT_レースキー_日",
        "OUT_レースキー_R",
        "OUT_馬番",
        "OUT_馬成績_着順",
        "OUT_馬成績_確定単勝オッズ",
    ]
]

# インプット (説明変数) と結合
df = pd.merge(
    df_input,
    df_output_subset,
    left_on=[
        "IN_レースキー_場コード",
        "IN_レースキー_年",
        "IN_レースキー_回",
        "IN_レースキー_日",
        "IN_レースキー_R",
        "IN_馬番",
    ],
    right_on=[
        "OUT_レースキー_場コード",
        "OUT_レースキー_年",
        "OUT_レースキー_回",
        "OUT_レースキー_日",
        "OUT_レースキー_R",
        "OUT_馬番",
    ],
    how="left",
    indicator=True,
)
df = df.drop(
    columns=[
        "OUT_レースキー_場コード",
        "OUT_レースキー_年",
        "OUT_レースキー_回",
        "OUT_レースキー_日",
        "OUT_レースキー_R",
        "OUT_馬番",
    ],
    axis=1,
)

# レースキーを作成
df["IN_レースキー"] = (
    df["IN_レースキー_場コード"]
    + df["IN_レースキー_年"]
    + df["IN_レースキー_回"]
    + df["IN_レースキー_日"]
    + df["IN_レースキー_R"]
)

# 数値に変換してソート
df["IN_年月日"] = pd.to_numeric(df["IN_年月日"], errors="coerce").astype(int)
df["IN_発走時間"] = pd.to_numeric(df["IN_発走時間"], errors="coerce").astype(int)
df["IN_馬番"] = pd.to_numeric(df["IN_馬番"], errors="coerce").astype(int)
df = df.sort_values(
    by=["IN_年月日", "IN_発走時間", "IN_馬番"], ascending=[False, True, True]
).reset_index(drop=True)

# 数値を文字列に戻す
df["IN_年月日"] = df["IN_年月日"].astype(str)
df["IN_発走時間"] = df["IN_発走時間"].apply(lambda x: str(x).zfill(4))
df["IN_馬番"] = df["IN_馬番"].apply(lambda x: str(x).zfill(2)).astype("category")

df_summary = summarize_object_columns(df)

save_file_path = os.path.join(str(file_directory), "ALL_DATA.feather")
df.to_feather(save_file_path)
