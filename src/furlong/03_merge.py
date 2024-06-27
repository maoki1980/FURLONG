import os

import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder
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


# dfに含まれるカテゴリ列をラベルエンコードする関数
def label_encode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    le = LabelEncoder()

    # カテゴリ型の列を検出し、ラベルエンコーディングを適用
    for column in df.select_dtypes(include=["category"]).columns:
        df[column] = le.fit_transform(df[column])

    return df


# 環境変数の読み込み
project_path = "../../"
env_file = os.getenv("ENV_FILE", os.path.join(project_path, ".env"))
load_dotenv(env_file)
file_directory = os.getenv("DF_DIR")
since_date = int(os.getenv("DF_SINCE"))
previous_data = 5  # 何走分のデータを含めるか

df_bac = read_df_file(str(file_directory), "BAC")
df_hjc = read_df_file(str(file_directory), "HJC")
df_kab = read_df_file(str(file_directory), "KAB")
df_kyi = read_df_file(str(file_directory), "KYI")
df_sed = read_df_file(str(file_directory), "SED")
df_ks = read_df_file(str(file_directory), "KS")
df_ukc = read_df_file(str(file_directory), "UKC")

df_ukc = df_ukc.sort_values(
    by=["血統登録番号", "データ年月日"], ascending=[True, False]
).reset_index(drop=True)
df_ukc = df_ukc.drop_duplicates(subset="血統登録番号", keep="first")
df_ukc = df_ukc.drop(columns=["データ年月日"], axis=1)

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
df_horse2 = pd.merge(
    df_horse,
    df_ukc,
    on=["血統登録番号"],
    how="left",
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

df_output2 = pd.merge(
    df_output,
    df_ks,
    left_on=[
        "騎手コード",
        "年月日",
    ],
    right_on=[
        "騎手コード",
        "データ年月日",
    ],
    how="left",
)
df_output2 = df_output2.drop(columns=["データ年月日"], axis=1)

df_input = merge_previous_race_data(df_horse2, df_output2, previous_data)

# カラム名に接頭辞を追加
prefix = "IN_"
df_input.columns = [prefix + col for col in df_input.columns]
prefix = "OUT_"
df_output2.columns = [prefix + col for col in df_output2.columns]

# アウトプット (目的変数) から必要な情報を抽出
df_output_subset = df_output2[
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

# データの日付による絞り込み
df["IN_年月日"] = pd.to_numeric(df["IN_年月日"], errors="coerce").astype(int)
df = df[df["IN_年月日"] >= since_date].reset_index(drop=True)

# 数値に変換してソート
df["IN_発走時間"] = pd.to_numeric(df["IN_発走時間"], errors="coerce").astype(int)
df = df.sort_values(
    by=["IN_年月日", "IN_発走時間", "IN_馬番"], ascending=[False, True, True]
).reset_index(drop=True)

# 数値を文字列に戻す
df["IN_年月日"] = df["IN_年月日"].astype(str)
df["IN_発走時間"] = df["IN_発走時間"].apply(lambda x: str(x).zfill(4))

# object型とcategory型の列の欠損値を空文字で補完する
fill_target_cols = df.select_dtypes(include=["object", "category"]).columns
for col in fill_target_cols:  # category型の列に空文字を追加
    if df[col].dtype.name == "category":
        df[col] = df[col].cat.add_categories(["NaN"])
df[fill_target_cols] = df[fill_target_cols].fillna("NaN")

# 数値列については補完方法辞書にしたがって補完する
fill_type_dic = {
    "馬番": 0,
    "馬成績_着順": "median",
    "馬成績_タイム": "mean",
    "馬成績_斤量": "mean",
    "馬成績_確定単勝オッズ": "median",
    "馬成績_確定単勝オッズ_log1p": "median",
    "馬成績_確定単勝人気順位": "median",
    "JRDBデータ_IDM": "mean",
    "JRDBデータ_素点": "mean",
    "JRDBデータ_馬場差": "mean",
    "JRDBデータ_テン指数_boxcox": "mean",
    "JRDBデータ_上がり指数_boxcox": "mean",
    "JRDBデータ_ペース指数_boxcox": "mean",
    "JRDBデータ_レースP指数_boxcox": "mean",
    "JRDBデータ_1(2)着タイム差": "mean",
    "JRDBデータ_前3Fタイム": "mean",
    "JRDBデータ_後3Fタイム": "mean",
    "確定複勝オッズ下_log1p": "median",
    "10時単勝オッズ_log1p": "median",
    "10時複勝オッズ_log1p": "median",
    "コーナー順位1": "median",
    "コーナー順位2": "median",
    "コーナー順位3": "median",
    "コーナー順位4": "median",
    "前3F先頭差": "mean",
    "後3F先頭差": "mean",
    "騎手コード": 0,
    "調教師コード": 0,
    "馬体重": "mean",
    "馬体重増減": "mean",
    "本賞金": 0,
    "収得賞金": 0,
    "レース条件_距離": "mean",
    "頭数": "median",
    "芝馬場差": "mean",
    "直線馬場差最内": "mean",
    "直線馬場差内": "mean",
    "直線馬場差中": "mean",
    "直線馬場差外": "mean",
    "直線馬場差大外": "mean",
    "ダ馬場差": "mean",
    "連続何日目": "median",
    "草丈": "mean",
    "中間降水量": "mean",
    "騎手生年": "median",
    "初免許年": "median",
    "本年平地成績_1着率": 0,
    "本年平地成績_2着以内率": 0,
    "本年平地成績_3着以内率": 0,
    "本年障害成績_1着率": 0,
    "本年障害成績_2着以内率": 0,
    "本年障害成績_3着以内率": 0,
    "昨年平地成績_1着率": 0,
    "昨年平地成績_2着以内率": 0,
    "昨年平地成績_3着以内率": 0,
    "昨年障害成績_1着率": 0,
    "昨年障害成績_2着以内率": 0,
    "昨年障害成績_3着以内率": 0,
    "通算平地成績_1着率": 0,
    "通算平地成績_2着以内率": 0,
    "通算平地成績_3着以内率": 0,
    "通算障害成績_1着率": 0,
    "通算障害成績_2着以内率": 0,
    "通算障害成績_3着以内率": 0,
}

# 辞書を元に補完する
for n in range(1, previous_data + 1):
    for key, value in fill_type_dic.items():
        column_name = f"IN_{n}走前_{key}"
        if column_name in df.columns:
            if value == "mean":
                average_value = df[column_name].mean(skipna=True)
                df[column_name].fillna(average_value, inplace=True)
            elif value == "median":
                average_value = df[column_name].median(skipna=True)
                df[column_name].fillna(average_value, inplace=True)
            else:
                df[column_name].fillna(value, inplace=True)

# dfのラベルエンコーディング
df = label_encode_dataframe(df)

# dfの要約用
df_summary = summarize_object_columns(df)

save_file_path = os.path.join(str(file_directory), "ALL_DATA.feather")
df.to_feather(save_file_path)
