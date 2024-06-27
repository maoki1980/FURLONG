import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import PowerTransformer
from utility import summarize_object_columns


# featherファイルを読み込む関数
def read_df_file(file_directory, file_category):
    file_path = os.path.join(file_directory, f"{file_category}.feather")
    df = pd.read_feather(file_path)
    df.replace("", np.nan, inplace=True)
    df = df.dropna(how="all").reset_index(drop=True)
    return df


# フォーマット済みdfをfeatherファイルに保存する関数
def save_df_file(file_directory, df, file_category):
    file_path = os.path.join(file_directory, f"{file_category}_fixed.feather")
    df.to_feather(file_path)


# 馬成績のタイムを秒に変換する関数
def convert_time_to_seconds(time_str):
    # 分と秒の部分を抽出
    minutes = int(time_str[0])
    seconds = int(time_str[1:])
    # 秒に変換
    return float(minutes * 60 + seconds / 10)


# 馬体重増減の値をフォーマットする関数
def format_weight_change(weight_str):
    if pd.isna(weight_str):
        return weight_str
    weight_str = weight_str.replace(" ", "")
    weight_str = weight_str.replace("+", "")
    return weight_str


# 成績の文字列をパースして着順の割合を出す関数
def parse_and_calculate_rates(s, n):
    # 桁数
    col = n * 4
    # 欠損値の場合は [np.nan, np.nan, np.nan] を返す
    if pd.isna(s):
        return [np.nan, np.nan, np.nan]
    # スペースを0に置換
    s = s.replace(" ", "0")
    # 固定長にする
    s = s.zfill(col)
    # n桁ずつ切り取ってリストに保存
    parts = [s[i : i + n] for i in range(0, col, n)]
    # 各部分を整数に変換
    parsed_parts = [int(part) for part in parts]

    # 1着、2着、3着、それ以外の回数
    first_place = parsed_parts[0]
    second_place = parsed_parts[1]
    third_place = parsed_parts[2]
    other_place = parsed_parts[3]
    total_races = first_place + second_place + third_place + other_place

    if total_races == 0:
        return [0.0, 0.0, 0.0]

    # 1着率
    first_rate = first_place / total_races
    # 2着以内の率
    top_two_rate = (first_place + second_place) / total_races
    # 3着以内の率
    top_three_rate = (first_place + second_place + third_place) / total_races

    return [first_rate, top_two_rate, top_three_rate]


# Box-Cox変換を行う関数
def boxcox_transform_column(df, column_name):
    # 列の最小値を確認しシフト
    min_value = df[column_name].min()
    if min_value <= 0:
        shift = abs(min_value) + 1
        df[column_name + "_shifted"] = df[column_name] + shift
    else:
        shift = 0
        df[column_name + "_shifted"] = df[column_name]

    # PowerTransformerのインスタンスを作成
    pt = PowerTransformer(method="box-cox")

    # 変換の適用 (2D配列を期待するためreshapeが必要)
    df[column_name + "_boxcox"] = pt.fit_transform(df[[column_name + "_shifted"]])

    # シフト量を返す
    return pt, shift


# Box-Cox変換を行う関数
def log1p_transform_column(series):
    min_value = series.min()
    shifted_series = series - min_value + 1
    log1p_transformed = np.log1p(shifted_series)
    return log1p_transformed


# 環境変数の読み込み
project_path = "../../"
env_file = os.getenv("ENV_FILE", os.path.join(project_path, ".env"))
load_dotenv(env_file)
file_directory = os.getenv("DF_DIR")


df_bac = read_df_file(str(file_directory), "BAC")


df_bac["レース条件_距離"] = pd.to_numeric(
    df_bac["レース条件_距離"], errors="coerce"
).astype(int)
df_bac["レース条件_芝ダ障害コード"] = df_bac["レース条件_芝ダ障害コード"].astype(
    "category"
)
df_bac["レース条件_右左"] = df_bac["レース条件_右左"].astype("category")
df_bac["レース条件_内外"] = df_bac["レース条件_内外"].astype("category")
df_bac["レース条件_種別"] = df_bac["レース条件_種別"].astype("category")
df_bac["レース条件_条件"] = df_bac["レース条件_条件"].astype("category")
df_bac["レース条件_記号1"] = df_bac["レース条件_記号"].str[0].astype("category")
df_bac["レース条件_記号2"] = df_bac["レース条件_記号"].str[1:3].astype("category")
df_bac["レース条件_記号3"] = df_bac["レース条件_記号"].str[3:].astype("category")
df_bac["レース条件_重量"] = df_bac["レース条件_重量"].astype("category")
df_bac["レース条件_グレード"] = (
    df_bac["レース条件_グレード"].fillna("0").astype("category")
)
df_bac["レース名"] = df_bac["レース名"].fillna("")
df_bac["回数"] = df_bac["回数"].fillna("")
df_bac["頭数"] = pd.to_numeric(df_bac["頭数"], errors="coerce").astype(int)
df_bac["コース"] = df_bac["コース"].fillna("0").astype("category")
df_bac = df_bac[
    [
        "レースキー_場コード",
        "レースキー_年",
        "レースキー_回",
        "レースキー_日",
        "レースキー_R",
        "年月日",
        "発走時間",
        "レース条件_距離",
        "レース条件_芝ダ障害コード",
        "レース条件_右左",
        "レース条件_内外",
        "レース条件_種別",
        "レース条件_条件",
        "レース条件_記号1",
        "レース条件_記号2",
        "レース条件_記号3",
        "レース条件_重量",
        "レース条件_グレード",
        "レース名",
        "回数",
        "頭数",
    ]
]

save_df_file(str(file_directory), df_bac, "BAC")
summary_bac = summarize_object_columns(df_bac)


df_hjc = read_df_file(str(file_directory), "HJC")


df_hjc["単勝払戻1_馬番"] = df_hjc["単勝払戻1_馬番"].astype("category")
df_hjc["単勝払戻1_払戻金"] = pd.to_numeric(
    df_hjc["単勝払戻1_払戻金"], errors="coerce"
).astype(int)
df_hjc["単勝払戻2_馬番"] = df_hjc["単勝払戻2_馬番"].astype("category")
df_hjc["単勝払戻2_払戻金"] = pd.to_numeric(
    df_hjc["単勝払戻2_払戻金"], errors="coerce"
).astype(int)
df_hjc["単勝払戻3_馬番"] = df_hjc["単勝払戻3_馬番"].astype("category")
df_hjc["単勝払戻3_払戻金"] = pd.to_numeric(
    df_hjc["単勝払戻3_払戻金"], errors="coerce"
).astype(int)
df_hjc["複勝払戻1_馬番"] = df_hjc["複勝払戻1_馬番"].astype("category")
df_hjc["複勝払戻1_払戻金"] = pd.to_numeric(
    df_hjc["複勝払戻1_払戻金"], errors="coerce"
).astype(int)
df_hjc["複勝払戻2_馬番"] = df_hjc["複勝払戻2_馬番"].astype("category")
df_hjc["複勝払戻2_払戻金"] = pd.to_numeric(
    df_hjc["複勝払戻2_払戻金"], errors="coerce"
).astype(int)
df_hjc["複勝払戻3_馬番"] = df_hjc["複勝払戻3_馬番"].astype("category")
df_hjc["複勝払戻3_払戻金"] = pd.to_numeric(
    df_hjc["複勝払戻3_払戻金"], errors="coerce"
).astype(int)
df_hjc["複勝払戻4_馬番"] = df_hjc["複勝払戻4_馬番"].astype("category")
df_hjc["複勝払戻4_払戻金"] = pd.to_numeric(
    df_hjc["複勝払戻4_払戻金"], errors="coerce"
).astype(int)
df_hjc["複勝払戻5_馬番"] = df_hjc["複勝払戻5_馬番"].astype("category")
df_hjc["複勝払戻5_払戻金"] = pd.to_numeric(
    df_hjc["複勝払戻5_払戻金"], errors="coerce"
).astype(int)
df_hjc["枠連払戻1_枠番組合せ"] = df_hjc["枠連払戻1_枠番組合せ"].astype("category")
df_hjc["枠連払戻1_払戻金"] = pd.to_numeric(
    df_hjc["枠連払戻1_払戻金"], errors="coerce"
).astype(int)
df_hjc["枠連払戻2_枠番組合せ"] = df_hjc["枠連払戻2_枠番組合せ"].astype("category")
df_hjc["枠連払戻2_払戻金"] = pd.to_numeric(
    df_hjc["枠連払戻2_払戻金"], errors="coerce"
).astype(int)
df_hjc["枠連払戻3_枠番組合せ"] = df_hjc["枠連払戻3_枠番組合せ"].astype("category")
df_hjc["枠連払戻3_払戻金"] = pd.to_numeric(
    df_hjc["枠連払戻3_払戻金"], errors="coerce"
).astype(int)
df_hjc["馬連払戻1_馬番組合せ"] = df_hjc["馬連払戻1_馬番組合せ"].astype("category")
df_hjc["馬連払戻1_払戻金"] = pd.to_numeric(
    df_hjc["馬連払戻1_払戻金"], errors="coerce"
).astype(int)
df_hjc["馬連払戻2_馬番組合せ"] = df_hjc["馬連払戻2_馬番組合せ"].astype("category")
df_hjc["馬連払戻2_払戻金"] = pd.to_numeric(
    df_hjc["馬連払戻2_払戻金"], errors="coerce"
).astype(int)
df_hjc["馬連払戻3_馬番組合せ"] = df_hjc["馬連払戻3_馬番組合せ"].astype("category")
df_hjc["馬連払戻3_払戻金"] = pd.to_numeric(
    df_hjc["馬連払戻3_払戻金"], errors="coerce"
).astype(int)
df_hjc["ワイド払戻1_馬番組合せ"] = df_hjc["ワイド払戻1_馬番組合せ"].astype("category")
df_hjc["ワイド払戻1_払戻金"] = pd.to_numeric(
    df_hjc["ワイド払戻1_払戻金"], errors="coerce"
).astype(int)
df_hjc["ワイド払戻2_馬番組合せ"] = df_hjc["ワイド払戻2_馬番組合せ"].astype("category")
df_hjc["ワイド払戻2_払戻金"] = pd.to_numeric(
    df_hjc["ワイド払戻2_払戻金"], errors="coerce"
).astype(int)
df_hjc["ワイド払戻3_馬番組合せ"] = df_hjc["ワイド払戻3_馬番組合せ"].astype("category")
df_hjc["ワイド払戻3_払戻金"] = pd.to_numeric(
    df_hjc["ワイド払戻3_払戻金"], errors="coerce"
).astype(int)
df_hjc["ワイド払戻4_馬番組合せ"] = df_hjc["ワイド払戻4_馬番組合せ"].astype("category")
df_hjc["ワイド払戻4_払戻金"] = pd.to_numeric(
    df_hjc["ワイド払戻4_払戻金"], errors="coerce"
).astype(int)
df_hjc["ワイド払戻5_馬番組合せ"] = df_hjc["ワイド払戻5_馬番組合せ"].astype("category")
df_hjc["ワイド払戻5_払戻金"] = pd.to_numeric(
    df_hjc["ワイド払戻5_払戻金"], errors="coerce"
).astype(int)
df_hjc["ワイド払戻6_馬番組合せ"] = df_hjc["ワイド払戻6_馬番組合せ"].astype("category")
df_hjc["ワイド払戻6_払戻金"] = pd.to_numeric(
    df_hjc["ワイド払戻6_払戻金"], errors="coerce"
).astype(int)
df_hjc["ワイド払戻7_馬番組合せ"] = df_hjc["ワイド払戻7_馬番組合せ"].astype("category")
df_hjc["ワイド払戻7_払戻金"] = pd.to_numeric(
    df_hjc["ワイド払戻7_払戻金"], errors="coerce"
).astype(int)
df_hjc = df_hjc[
    [
        "レースキー_場コード",
        "レースキー_年",
        "レースキー_回",
        "レースキー_日",
        "レースキー_R",
        "単勝払戻1_馬番",
        "単勝払戻1_払戻金",
        "単勝払戻2_馬番",
        "単勝払戻2_払戻金",
        "単勝払戻3_馬番",
        "単勝払戻3_払戻金",
        "複勝払戻1_馬番",
        "複勝払戻1_払戻金",
        "複勝払戻2_馬番",
        "複勝払戻2_払戻金",
        "複勝払戻3_馬番",
        "複勝払戻3_払戻金",
        "複勝払戻4_馬番",
        "複勝払戻4_払戻金",
        "複勝払戻5_馬番",
        "複勝払戻5_払戻金",
        "枠連払戻1_枠番組合せ",
        "枠連払戻1_払戻金",
        "枠連払戻2_枠番組合せ",
        "枠連払戻2_払戻金",
        "枠連払戻3_枠番組合せ",
        "枠連払戻3_払戻金",
        "馬連払戻1_馬番組合せ",
        "馬連払戻1_払戻金",
        "馬連払戻2_馬番組合せ",
        "馬連払戻2_払戻金",
        "馬連払戻3_馬番組合せ",
        "馬連払戻3_払戻金",
        "ワイド払戻1_馬番組合せ",
        "ワイド払戻1_払戻金",
        "ワイド払戻2_馬番組合せ",
        "ワイド払戻2_払戻金",
        "ワイド払戻3_馬番組合せ",
        "ワイド払戻3_払戻金",
        "ワイド払戻4_馬番組合せ",
        "ワイド払戻4_払戻金",
        "ワイド払戻5_馬番組合せ",
        "ワイド払戻5_払戻金",
        "ワイド払戻6_馬番組合せ",
        "ワイド払戻6_払戻金",
        "ワイド払戻7_馬番組合せ",
        "ワイド払戻7_払戻金",
    ]
]

save_df_file(str(file_directory), df_hjc, "HJC")
summary_hjc = summarize_object_columns(df_hjc)


df_kab = read_df_file(str(file_directory), "KAB")


df_kab["曜日"] = df_kab["曜日"].astype("category")
df_kab["天候コード"] = df_kab["天候コード"].fillna("0").astype("category")
df_kab["芝馬場状態コード"] = df_kab["芝馬場状態コード"].astype("category")
df_kab["芝馬場状態内"] = df_kab["芝馬場状態内"].fillna("0").astype("category")
df_kab["芝馬場状態中"] = df_kab["芝馬場状態中"].fillna("0").astype("category")
df_kab["芝馬場状態外"] = df_kab["芝馬場状態外"].fillna("0").astype("category")

df_kab["芝馬場差"] = pd.to_numeric(df_kab["芝馬場差"], errors="coerce")
average_value = df_kab["芝馬場差"].mean()
df_kab["芝馬場差"] = df_kab["芝馬場差"].fillna(average_value).astype(int)

df_kab["直線馬場差最内"] = pd.to_numeric(df_kab["直線馬場差最内"], errors="coerce")
average_value = df_kab["直線馬場差最内"].mean()
df_kab["直線馬場差最内"] = df_kab["直線馬場差最内"].fillna(average_value).astype(int)

df_kab["直線馬場差内"] = pd.to_numeric(df_kab["直線馬場差内"], errors="coerce")
average_value = df_kab["直線馬場差内"].mean()
df_kab["直線馬場差内"] = df_kab["直線馬場差内"].fillna(average_value).astype(int)

df_kab["直線馬場差中"] = pd.to_numeric(df_kab["直線馬場差中"], errors="coerce")
average_value = df_kab["直線馬場差中"].mean()
df_kab["直線馬場差中"] = df_kab["直線馬場差中"].fillna(average_value).astype(int)

df_kab["直線馬場差外"] = pd.to_numeric(df_kab["直線馬場差外"], errors="coerce")
average_value = df_kab["直線馬場差外"].mean()
df_kab["直線馬場差外"] = df_kab["直線馬場差外"].fillna(average_value).astype(int)

df_kab["直線馬場差大外"] = pd.to_numeric(df_kab["直線馬場差大外"], errors="coerce")
average_value = df_kab["直線馬場差大外"].mean()
df_kab["直線馬場差大外"] = df_kab["直線馬場差大外"].fillna(average_value).astype(int)

df_kab["ダ馬場状態コード"] = df_kab["ダ馬場状態コード"].fillna("0").astype("category")
df_kab["ダ馬場状態内"] = df_kab["ダ馬場状態内"].fillna("0").astype("category")
df_kab["ダ馬場状態中"] = df_kab["ダ馬場状態中"].fillna("0").astype("category")
df_kab["ダ馬場状態外"] = df_kab["ダ馬場状態外"].fillna("0").astype("category")

df_kab["ダ馬場差"] = pd.to_numeric(df_kab["ダ馬場差"], errors="coerce")
average_value = df_kab["ダ馬場差"].mean()
df_kab["ダ馬場差"] = df_kab["ダ馬場差"].fillna(average_value).astype(int)

df_kab["連続何日目"] = pd.to_numeric(df_kab["連続何日目"], errors="coerce")
average_value = df_kab["連続何日目"].median()
df_kab["連続何日目"] = df_kab["連続何日目"].fillna(average_value).astype(int)

df_kab["芝種類"] = df_kab["芝種類"].fillna("0").astype("category")

df_kab["草丈"] = pd.to_numeric(df_kab["草丈"], errors="coerce")
average_value = df_kab["草丈"].mean()
df_kab["草丈"] = df_kab["草丈"].fillna(average_value).astype(float)

df_kab["転圧"] = df_kab["転圧"].fillna("0").astype("category")
df_kab["凍結防止剤"] = df_kab["凍結防止剤"].fillna("0").astype("category")

df_kab["中間降水量"] = pd.to_numeric(df_kab["中間降水量"], errors="coerce")
average_value = df_kab["中間降水量"].mean()
df_kab["中間降水量"] = df_kab["中間降水量"].fillna(average_value).astype(float)

df_kab = df_kab[
    [
        "開催キー_場コード",
        "開催キー_年",
        "開催キー_回",
        "開催キー_日",
        "年月日",
        "曜日",
        "芝馬場状態コード",
        "芝馬場状態内",
        "芝馬場状態中",
        "芝馬場状態外",
        "芝馬場差",
        "直線馬場差最内",
        "直線馬場差内",
        "直線馬場差中",
        "直線馬場差外",
        "直線馬場差大外",
        "ダ馬場状態コード",
        "ダ馬場状態内",
        "ダ馬場状態中",
        "ダ馬場状態外",
        "ダ馬場差",
        "連続何日目",
        "芝種類",
        "草丈",
        "転圧",
        "凍結防止剤",
        "中間降水量",
    ]
]

save_df_file(str(file_directory), df_kab, "KAB")
summary_kab = summarize_object_columns(df_kab)


df_kyi = read_df_file(str(file_directory), "KYI")


df_kyi["馬番"] = df_kyi["馬番"].fillna(0).astype(int)
df_kyi["血統登録番号"] = df_kyi["血統登録番号"].fillna(0).astype(int)

df_kyi["IDM"] = pd.to_numeric(df_kyi["IDM"], errors="coerce")
average_value = df_kyi["IDM"].median()
df_kyi["IDM"] = df_kyi["IDM"].fillna(average_value).astype(float)
boxcox_transform_column(df_kyi, "IDM")

df_kyi["騎手指数"] = pd.to_numeric(df_kyi["騎手指数"], errors="coerce")
average_value = df_kyi["騎手指数"].median()
df_kyi["騎手指数"] = df_kyi["騎手指数"].fillna(average_value).astype(float)
df_kyi["騎手指数_log1p"] = log1p_transform_column(df_kyi["騎手指数"])

df_kyi["情報指数"] = pd.to_numeric(df_kyi["情報指数"], errors="coerce")
average_value = df_kyi["情報指数"].median()
df_kyi["情報指数"] = df_kyi["情報指数"].fillna(average_value).astype(float)
df_kyi["情報指数_log1p"] = log1p_transform_column(df_kyi["情報指数"])

df_kyi["総合指数"] = pd.to_numeric(df_kyi["総合指数"], errors="coerce")
average_value = df_kyi["総合指数"].median()
df_kyi["総合指数"] = df_kyi["総合指数"].fillna(average_value).astype(float)

df_kyi["脚質"] = df_kyi["脚質"].fillna("0").astype("category")
df_kyi["距離適性"] = df_kyi["距離適性"].fillna("0").astype("category")
df_kyi["上昇度"] = df_kyi["上昇度"].fillna("0").astype("category")

df_kyi["ローテーション"] = pd.to_numeric(df_kyi["ローテーション"], errors="coerce")
average_value = df_kyi["ローテーション"].median()
df_kyi["ローテーション"] = df_kyi["ローテーション"].fillna(average_value).astype(int)

df_kyi["基準オッズ"] = pd.to_numeric(df_kyi["基準オッズ"], errors="coerce")
average_value = df_kyi["基準オッズ"].mean()
df_kyi["基準オッズ"] = df_kyi["基準オッズ"].fillna(average_value).astype(float)
df_kyi["基準オッズ_log1p"] = log1p_transform_column(df_kyi["基準オッズ"])

df_kyi["基準人気順位"] = pd.to_numeric(df_kyi["基準人気順位"], errors="coerce")
average_value = df_kyi["基準人気順位"].mean()
df_kyi["基準人気順位"] = df_kyi["基準人気順位"].fillna(average_value).astype(int)

df_kyi["基準複勝オッズ"] = pd.to_numeric(df_kyi["基準複勝オッズ"], errors="coerce")
average_value = df_kyi["基準複勝オッズ"].mean()
df_kyi["基準複勝オッズ"] = df_kyi["基準複勝オッズ"].fillna(average_value).astype(float)
df_kyi["基準複勝オッズ_log1p"] = log1p_transform_column(df_kyi["基準複勝オッズ"])

df_kyi["基準複勝人気順位"] = pd.to_numeric(df_kyi["基準複勝人気順位"], errors="coerce")
average_value = df_kyi["基準複勝人気順位"].mean()
df_kyi["基準複勝人気順位"] = (
    df_kyi["基準複勝人気順位"].fillna(average_value).astype(int)
)

df_kyi["特定情報◎"] = (
    pd.to_numeric(df_kyi["特定情報◎"], errors="coerce").fillna(0).astype(int)
)
df_kyi["特定情報◎_log1p"] = log1p_transform_column(df_kyi["特定情報◎"])

df_kyi["特定情報○"] = (
    pd.to_numeric(df_kyi["特定情報○"], errors="coerce").fillna(0).astype(int)
)
df_kyi["特定情報○_log1p"] = log1p_transform_column(df_kyi["特定情報○"])

df_kyi["特定情報▲"] = (
    pd.to_numeric(df_kyi["特定情報▲"], errors="coerce").fillna(0).astype(int)
)
df_kyi["特定情報▲_log1p"] = log1p_transform_column(df_kyi["特定情報▲"])

df_kyi["特定情報△"] = (
    pd.to_numeric(df_kyi["特定情報△"], errors="coerce").fillna(0).astype(int)
)
df_kyi["特定情報△_log1p"] = log1p_transform_column(df_kyi["特定情報△"])

df_kyi["特定情報×"] = (
    pd.to_numeric(df_kyi["特定情報×"], errors="coerce").fillna(0).astype(int)
)
df_kyi["特定情報×_log1p"] = log1p_transform_column(df_kyi["特定情報×"])

df_kyi["総合情報◎"] = (
    pd.to_numeric(df_kyi["総合情報◎"], errors="coerce").fillna(0).astype(int)
)
df_kyi["総合情報◎_log1p"] = log1p_transform_column(df_kyi["総合情報◎"])
boxcox_transform_column(df_kyi, "総合情報◎_log1p")

df_kyi["総合情報○"] = (
    pd.to_numeric(df_kyi["総合情報○"], errors="coerce").fillna(0).astype(int)
)
df_kyi["総合情報○_log1p"] = log1p_transform_column(df_kyi["総合情報○"])
boxcox_transform_column(df_kyi, "総合情報○_log1p")

df_kyi["総合情報▲"] = (
    pd.to_numeric(df_kyi["総合情報▲"], errors="coerce").fillna(0).astype(int)
)
df_kyi["総合情報▲_log1p"] = log1p_transform_column(df_kyi["総合情報▲"])
boxcox_transform_column(df_kyi, "総合情報▲_log1p")

df_kyi["総合情報△"] = (
    pd.to_numeric(df_kyi["総合情報△"], errors="coerce").fillna(0).astype(int)
)
df_kyi["総合情報△_log1p"] = log1p_transform_column(df_kyi["総合情報△"])
boxcox_transform_column(df_kyi, "総合情報△_log1p")

df_kyi["総合情報×"] = (
    pd.to_numeric(df_kyi["総合情報×"], errors="coerce").fillna(0).astype(int)
)
df_kyi["総合情報×_log1p"] = log1p_transform_column(df_kyi["総合情報×"])
boxcox_transform_column(df_kyi, "総合情報×_log1p")

df_kyi["人気指数"] = pd.to_numeric(df_kyi["人気指数"], errors="coerce")
average_value = df_kyi["人気指数"].median()
df_kyi["人気指数"] = df_kyi["人気指数"].fillna(average_value).astype(int)
df_kyi["人気指数_log1p"] = log1p_transform_column(df_kyi["人気指数"])
boxcox_transform_column(df_kyi, "人気指数_log1p")


df_kyi["調教指数"] = pd.to_numeric(df_kyi["調教指数"], errors="coerce")
average_value = df_kyi["調教指数"].median()
df_kyi["調教指数"] = df_kyi["調教指数"].fillna(average_value).astype(float)
boxcox_transform_column(df_kyi, "調教指数")

df_kyi["厩舎指数"] = pd.to_numeric(df_kyi["厩舎指数"], errors="coerce")
average_value = df_kyi["厩舎指数"].median()
df_kyi["厩舎指数"] = df_kyi["厩舎指数"].fillna(average_value).astype(float)

df_kyi["調教矢印コード"] = df_kyi["調教矢印コード"].fillna("0").astype("category")
df_kyi["厩舎評価コード"] = df_kyi["厩舎評価コード"].fillna("0").astype("category")

df_kyi["騎手期待連対率"] = pd.to_numeric(df_kyi["騎手期待連対率"], errors="coerce")
average_value = df_kyi["騎手期待連対率"].median()
df_kyi["騎手期待連対率"] = df_kyi["騎手期待連対率"].fillna(average_value).astype(float)
df_kyi["騎手期待連対率_log1p"] = log1p_transform_column(df_kyi["騎手期待連対率"])
boxcox_transform_column(df_kyi, "騎手期待連対率_log1p")

df_kyi["激走指数"] = pd.to_numeric(df_kyi["激走指数"], errors="coerce")
average_value = df_kyi["激走指数"].median()
df_kyi["激走指数"] = df_kyi["激走指数"].fillna(average_value).astype(int)

df_kyi["蹄コード"] = df_kyi["蹄コード"].fillna("00").astype("category")
df_kyi["重適正コード"] = df_kyi["重適正コード"].fillna("0").astype("category")
df_kyi["クラスコード"] = df_kyi["クラスコード"].fillna("00").astype("category")
df_kyi["ブリンカー"] = df_kyi["ブリンカー"].fillna("0").astype("category")

df_kyi["負担重量"] = pd.to_numeric(df_kyi["負担重量"], errors="coerce")
average_value = df_kyi["負担重量"].mean()
df_kyi["負担重量"] = (
    df_kyi["負担重量"].fillna(average_value).astype(float) * 0.1
)  # 0.1kg単位からkg単位に変換

df_kyi["見習い区分"] = df_kyi["見習い区分"].fillna("0").astype("category")
df_kyi["調教師所属"] = df_kyi["調教師所属"].fillna("").astype("category")
df_kyi["他データリンク用キー_前走1競走成績キー"] = df_kyi[
    "他データリンク用キー_前走1競走成績キー"
].fillna("")
df_kyi["他データリンク用キー_前走2競走成績キー"] = df_kyi[
    "他データリンク用キー_前走2競走成績キー"
].fillna("")
df_kyi["他データリンク用キー_前走3競走成績キー"] = df_kyi[
    "他データリンク用キー_前走3競走成績キー"
].fillna("")
df_kyi["他データリンク用キー_前走4競走成績キー"] = df_kyi[
    "他データリンク用キー_前走4競走成績キー"
].fillna("")
df_kyi["他データリンク用キー_前走5競走成績キー"] = df_kyi[
    "他データリンク用キー_前走5競走成績キー"
].fillna("")
df_kyi["枠番"] = df_kyi["枠番"].fillna("0").astype("category")
df_kyi["印コード_総合印"] = df_kyi["印コード_総合印"].fillna("0").astype("category")
df_kyi["印コード_IDM印"] = df_kyi["印コード_IDM印"].fillna("0").astype("category")
df_kyi["印コード_情報印"] = df_kyi["印コード_情報印"].fillna("0").astype("category")
df_kyi["印コード_騎手印"] = df_kyi["印コード_騎手印"].fillna("0").astype("category")
df_kyi["印コード_厩舎印"] = df_kyi["印コード_厩舎印"].fillna("0").astype("category")
df_kyi["印コード_調教印"] = df_kyi["印コード_調教印"].fillna("0").astype("category")
df_kyi["印コード_激走印"] = df_kyi["印コード_激走印"].fillna("0").astype("category")
df_kyi["芝適性コード"] = df_kyi["芝適性コード"].fillna("0").astype("category")
df_kyi["ダ適性コード"] = df_kyi["ダ適性コード"].fillna("0").astype("category")
df_kyi["騎手コード"] = df_kyi["騎手コード"].astype(int)
df_kyi["調教師コード"] = df_kyi["調教師コード"].fillna(0).astype(int)

df_kyi["賞金情報_獲得賞金"] = (
    pd.to_numeric(df_kyi["賞金情報_獲得賞金"], errors="coerce").fillna(0).astype(int)
)
df_kyi["賞金情報_獲得賞金_log1p"] = log1p_transform_column(df_kyi["賞金情報_獲得賞金"])

df_kyi["賞金情報_収得賞金"] = (
    pd.to_numeric(df_kyi["賞金情報_収得賞金"], errors="coerce").fillna(0).astype(int)
)
df_kyi["賞金情報_収得賞金_log1p"] = log1p_transform_column(df_kyi["賞金情報_収得賞金"])

df_kyi["賞金情報_条件クラス"] = (
    df_kyi["賞金情報_条件クラス"].fillna("0").astype("category")
)
df_kyi["距離適性2"] = df_kyi["距離適性2"].fillna("0").astype("category")
df_kyi["取消フラグ"] = df_kyi["取消フラグ"].fillna("0").astype("category")

df_kyi["激走順位"] = pd.to_numeric(df_kyi["激走順位"], errors="coerce")
average_value = df_kyi["激走順位"].mean()
df_kyi["激走順位"] = df_kyi["激走順位"].fillna(average_value).astype(int)

df_kyi["LS指数順位"] = pd.to_numeric(df_kyi["LS指数順位"], errors="coerce")
average_value = df_kyi["LS指数順位"].mean()
df_kyi["LS指数順位"] = df_kyi["LS指数順位"].fillna(average_value).astype(int)

df_kyi["テン指数順位"] = pd.to_numeric(df_kyi["テン指数順位"], errors="coerce")
average_value = df_kyi["テン指数順位"].mean()
df_kyi["テン指数順位"] = df_kyi["テン指数順位"].fillna(average_value).astype(int)

df_kyi["ペース指数順位"] = pd.to_numeric(df_kyi["ペース指数順位"], errors="coerce")
average_value = df_kyi["ペース指数順位"].mean()
df_kyi["ペース指数順位"] = df_kyi["ペース指数順位"].fillna(average_value).astype(int)

df_kyi["上がり指数順位"] = pd.to_numeric(df_kyi["上がり指数順位"], errors="coerce")
average_value = df_kyi["上がり指数順位"].mean()
df_kyi["上がり指数順位"] = df_kyi["上がり指数順位"].fillna(average_value).astype(int)

df_kyi["位置指数順位"] = pd.to_numeric(df_kyi["位置指数順位"], errors="coerce")
average_value = df_kyi["位置指数順位"].mean()
df_kyi["位置指数順位"] = df_kyi["位置指数順位"].fillna(average_value).astype(int)

df_kyi["騎手期待単勝率"] = pd.to_numeric(df_kyi["騎手期待単勝率"], errors="coerce")
average_value = df_kyi["騎手期待単勝率"].median()
df_kyi["騎手期待単勝率"] = df_kyi["騎手期待単勝率"].fillna(average_value).astype(float)
boxcox_transform_column(df_kyi, "騎手期待単勝率")

df_kyi["騎手期待3着内率"] = pd.to_numeric(df_kyi["騎手期待3着内率"], errors="coerce")
average_value = df_kyi["騎手期待3着内率"].median()
df_kyi["騎手期待3着内率"] = (
    df_kyi["騎手期待3着内率"].fillna(average_value).astype(float)
)
boxcox_transform_column(df_kyi, "騎手期待3着内率")

df_kyi["輸送区分"] = df_kyi["輸送区分"].fillna("0").astype("category")

df_kyi["万券指数"] = pd.to_numeric(df_kyi["万券指数"], errors="coerce")
average_value = df_kyi["万券指数"].median()
df_kyi["万券指数"] = df_kyi["万券指数"].fillna(average_value).astype(int)

df_kyi["万券印"] = df_kyi["万券印"].fillna("0").astype("category")
df_kyi["降級フラグ"] = df_kyi["降級フラグ"].fillna("").astype("category")
df_kyi["激走タイプ"] = df_kyi["激走タイプ"].fillna("").astype("category")

df_kyi["入厩何走目"] = pd.to_numeric(df_kyi["入厩何走目"], errors="coerce")
average_value = df_kyi["入厩何走目"].median()
df_kyi["入厩何走目"] = df_kyi["入厩何走目"].fillna(average_value).astype(int)
df_kyi["入厩何走目_log1p"] = log1p_transform_column(df_kyi["入厩何走目"])

df_kyi["放牧先ランク"] = df_kyi["放牧先ランク"].fillna("").astype("category")
df_kyi["厩舎ランク"] = df_kyi["厩舎ランク"].fillna("").astype("category")

df_kyi = df_kyi[
    [
        "レースキー_場コード",
        "レースキー_年",
        "レースキー_回",
        "レースキー_日",
        "レースキー_R",
        "馬番",
        "血統登録番号",
        "馬名",
        "IDM_boxcox",
        "騎手指数_log1p",
        "情報指数_log1p",
        "総合指数",
        "脚質",
        "距離適性",
        "上昇度",
        "ローテーション",
        "基準オッズ_log1p",
        "基準人気順位",
        "基準複勝オッズ_log1p",
        "基準複勝人気順位",
        "特定情報◎_log1p",
        "特定情報○_log1p",
        "特定情報▲_log1p",
        "特定情報△_log1p",
        "特定情報×_log1p",
        "総合情報◎_log1p_boxcox",
        "総合情報○_log1p_boxcox",
        "総合情報▲_log1p_boxcox",
        "総合情報△_log1p_boxcox",
        "総合情報×_log1p_boxcox",
        "人気指数_log1p_boxcox",
        "調教指数_boxcox",
        "厩舎指数",
        "調教矢印コード",
        "厩舎評価コード",
        "騎手期待連対率_log1p_boxcox",
        "激走指数",
        "蹄コード",
        "重適正コード",
        "クラスコード",
        "ブリンカー",
        "騎手名",
        "負担重量",
        "見習い区分",
        "調教師所属",
        "他データリンク用キー_前走1競走成績キー",
        "他データリンク用キー_前走2競走成績キー",
        "他データリンク用キー_前走3競走成績キー",
        "他データリンク用キー_前走4競走成績キー",
        "他データリンク用キー_前走5競走成績キー",
        "枠番",
        "印コード_総合印",
        "印コード_IDM印",
        "印コード_情報印",
        "印コード_騎手印",
        "印コード_厩舎印",
        "印コード_調教印",
        "印コード_激走印",
        "芝適性コード",
        "ダ適性コード",
        "騎手コード",
        "調教師コード",
        "賞金情報_獲得賞金_log1p",
        "賞金情報_収得賞金_log1p",
        "賞金情報_条件クラス",
        "距離適性2",
        "取消フラグ",
        "激走順位",
        "LS指数順位",
        "テン指数順位",
        "ペース指数順位",
        "上がり指数順位",
        "位置指数順位",
        "騎手期待単勝率_boxcox",
        "騎手期待3着内率_boxcox",
        "輸送区分",
        "万券指数",
        "万券印",
        "降級フラグ",
        "激走タイプ",
        "入厩何走目_log1p",
        "放牧先ランク",
        "厩舎ランク",
    ]
]

save_df_file(str(file_directory), df_kyi, "KYI")
summary_kyi = summarize_object_columns(df_kyi)


df_sed = read_df_file(str(file_directory), "SED")


df_sed["馬番"] = df_sed["馬番"].fillna(0).astype(int)
df_sed["競走成績キー"] = df_sed["競走成績キー_血統登録番号"].fillna("") + df_sed[
    "競走成績キー_年月日"
].fillna("")
df_sed["レース条件_馬場状態"] = (
    df_sed["レース条件_馬場状態"].fillna("00").astype("category")
)
df_sed["馬成績_着順"] = (
    pd.to_numeric(df_sed["馬成績_着順"], errors="coerce").fillna("99").astype(int)
)
df_sed["馬成績_異常区分"] = df_sed["馬成績_異常区分"].fillna("0").astype("category")
df_sed["馬成績_タイム"] = (
    df_sed["馬成績_タイム"].fillna("9999").apply(convert_time_to_seconds)
)
df_sed["馬成績_斤量"] = (
    pd.to_numeric(df_sed["馬成績_斤量"], errors="coerce").fillna(0).astype(float) * 0.1
)  # 0.1kg単位からkg単位に変換

df_sed["馬成績_確定単勝オッズ"] = pd.to_numeric(
    df_sed["馬成績_確定単勝オッズ"], errors="coerce"
)
average_value = df_sed["馬成績_確定単勝オッズ"].mean()
df_sed["馬成績_確定単勝オッズ"] = (
    df_sed["馬成績_確定単勝オッズ"].fillna(average_value).astype(float)
)
df_sed["馬成績_確定単勝オッズ_log1p"] = log1p_transform_column(
    df_sed["馬成績_確定単勝オッズ"]
)

df_sed["馬成績_確定単勝人気順位"] = pd.to_numeric(
    df_sed["馬成績_確定単勝人気順位"], errors="coerce"
)
average_value = df_sed["馬成績_確定単勝人気順位"].mean()
df_sed["馬成績_確定単勝人気順位"] = (
    df_sed["馬成績_確定単勝人気順位"].fillna(average_value).astype(int)
)

df_sed["JRDBデータ_IDM"] = pd.to_numeric(df_sed["JRDBデータ_IDM"], errors="coerce")
average_value = df_sed["JRDBデータ_IDM"].median()
df_sed["JRDBデータ_IDM"] = df_sed["JRDBデータ_IDM"].fillna(average_value).astype(float)

df_sed["JRDBデータ_素点"] = pd.to_numeric(df_sed["JRDBデータ_素点"], errors="coerce")
average_value = df_sed["JRDBデータ_素点"].median()
df_sed["JRDBデータ_素点"] = df_sed["JRDBデータ_素点"].fillna(average_value).astype(int)

df_sed["JRDBデータ_馬場差"] = pd.to_numeric(
    df_sed["JRDBデータ_馬場差"], errors="coerce"
)
average_value = df_sed["JRDBデータ_馬場差"].mean()
df_sed["JRDBデータ_馬場差"] = (
    df_sed["JRDBデータ_馬場差"].fillna(average_value).astype(int)
)

df_sed["JRDBデータ_コース取り"] = (
    df_sed["JRDBデータ_コース取り"].fillna("0").astype("category")
)
df_sed["JRDBデータ_上昇度コード"] = (
    df_sed["JRDBデータ_上昇度コード"].fillna("0").astype("category")
)
df_sed["JRDBデータ_クラスコード"] = (
    df_sed["JRDBデータ_クラスコード"].fillna("00").astype("category")
)
df_sed["JRDBデータ_馬体コード"] = (
    df_sed["JRDBデータ_馬体コード"].fillna("0").astype("category")
)
df_sed["JRDBデータ_気配コード"] = (
    df_sed["JRDBデータ_気配コード"].fillna("0").astype("category")
)
df_sed["JRDBデータ_レースペース"] = (
    df_sed["JRDBデータ_レースペース"].fillna("").astype("category")
)
df_sed["JRDBデータ_馬ペース"] = (
    df_sed["JRDBデータ_馬ペース"].fillna("").astype("category")
)

df_sed["JRDBデータ_テン指数"] = pd.to_numeric(
    df_sed["JRDBデータ_テン指数"], errors="coerce"
)
average_value = df_sed["JRDBデータ_テン指数"].median()
df_sed["JRDBデータ_テン指数"] = (
    df_sed["JRDBデータ_テン指数"].fillna(average_value).astype(float)
)
boxcox_transform_column(df_sed, "JRDBデータ_テン指数")

df_sed["JRDBデータ_上がり指数"] = pd.to_numeric(
    df_sed["JRDBデータ_上がり指数"], errors="coerce"
)
average_value = df_sed["JRDBデータ_上がり指数"].median()
df_sed["JRDBデータ_上がり指数"] = (
    df_sed["JRDBデータ_上がり指数"].fillna(average_value).astype(float)
)
boxcox_transform_column(df_sed, "JRDBデータ_上がり指数")

df_sed["JRDBデータ_ペース指数"] = pd.to_numeric(
    df_sed["JRDBデータ_ペース指数"], errors="coerce"
)
average_value = df_sed["JRDBデータ_ペース指数"].median()
df_sed["JRDBデータ_ペース指数"] = (
    df_sed["JRDBデータ_ペース指数"].fillna(average_value).astype(float)
)
boxcox_transform_column(df_sed, "JRDBデータ_ペース指数")

df_sed["JRDBデータ_レースP指数"] = pd.to_numeric(
    df_sed["JRDBデータ_レースP指数"], errors="coerce"
)
average_value = df_sed["JRDBデータ_レースP指数"].median()
df_sed["JRDBデータ_レースP指数"] = (
    df_sed["JRDBデータ_レースP指数"].fillna(average_value).astype(float)
)
boxcox_transform_column(df_sed, "JRDBデータ_レースP指数")

df_sed["JRDBデータ_1(2)着タイム差"] = pd.to_numeric(
    df_sed["JRDBデータ_1(2)着タイム差"], errors="coerce"
)
average_value = df_sed["JRDBデータ_1(2)着タイム差"].mean()
df_sed["JRDBデータ_1(2)着タイム差"] = (
    df_sed["JRDBデータ_1(2)着タイム差"].fillna(average_value).astype(float)
)

df_sed["JRDBデータ_前3Fタイム"] = pd.to_numeric(
    df_sed["JRDBデータ_前3Fタイム"], errors="coerce"
)
average_value = df_sed["JRDBデータ_前3Fタイム"].mean()
df_sed["JRDBデータ_前3Fタイム"] = (
    df_sed["JRDBデータ_前3Fタイム"].fillna(average_value).astype(float)
)

df_sed["JRDBデータ_後3Fタイム"] = pd.to_numeric(
    df_sed["JRDBデータ_後3Fタイム"], errors="coerce"
)
average_value = df_sed["JRDBデータ_後3Fタイム"].mean()
df_sed["JRDBデータ_後3Fタイム"] = (
    df_sed["JRDBデータ_後3Fタイム"].fillna(average_value).astype(float)
)

df_sed["確定複勝オッズ下"] = pd.to_numeric(df_sed["確定複勝オッズ下"], errors="coerce")
average_value = df_sed["確定複勝オッズ下"].mean()
df_sed["確定複勝オッズ下"] = (
    df_sed["確定複勝オッズ下"].fillna(average_value).astype(float)
)
df_sed["確定複勝オッズ下_log1p"] = log1p_transform_column(df_sed["確定複勝オッズ下"])

df_sed["10時単勝オッズ"] = pd.to_numeric(df_sed["10時単勝オッズ"], errors="coerce")
average_value = df_sed["10時単勝オッズ"].mean()
df_sed["10時単勝オッズ"] = df_sed["10時単勝オッズ"].fillna(average_value).astype(float)
df_sed["10時単勝オッズ_log1p"] = log1p_transform_column(df_sed["10時単勝オッズ"])

df_sed["10時複勝オッズ"] = pd.to_numeric(df_sed["10時複勝オッズ"], errors="coerce")
average_value = df_sed["10時複勝オッズ"].mean()
df_sed["10時複勝オッズ"] = df_sed["10時複勝オッズ"].fillna(average_value).astype(float)
df_sed["10時複勝オッズ_log1p"] = log1p_transform_column(df_sed["10時複勝オッズ"])

df_sed["コーナー順位1"] = pd.to_numeric(df_sed["コーナー順位1"], errors="coerce")
average_value = df_sed["コーナー順位1"].mean()
df_sed["コーナー順位1"] = df_sed["コーナー順位1"].fillna(average_value).astype(int)

df_sed["コーナー順位2"] = pd.to_numeric(df_sed["コーナー順位2"], errors="coerce")
average_value = df_sed["コーナー順位2"].mean()
df_sed["コーナー順位2"] = df_sed["コーナー順位2"].fillna(average_value).astype(int)

df_sed["コーナー順位3"] = pd.to_numeric(df_sed["コーナー順位3"], errors="coerce")
average_value = df_sed["コーナー順位3"].mean()
df_sed["コーナー順位3"] = df_sed["コーナー順位3"].fillna(average_value).astype(int)

df_sed["コーナー順位4"] = pd.to_numeric(df_sed["コーナー順位4"], errors="coerce")
average_value = df_sed["コーナー順位4"].mean()
df_sed["コーナー順位4"] = df_sed["コーナー順位4"].fillna(average_value).astype(int)

df_sed["前3F先頭差"] = pd.to_numeric(df_sed["前3F先頭差"], errors="coerce")
average_value = df_sed["前3F先頭差"].mean()
df_sed["前3F先頭差"] = df_sed["前3F先頭差"].fillna(average_value).astype(int)

df_sed["後3F先頭差"] = pd.to_numeric(df_sed["後3F先頭差"], errors="coerce")
average_value = df_sed["後3F先頭差"].mean()
df_sed["後3F先頭差"] = df_sed["後3F先頭差"].fillna(average_value).astype(int)

df_sed["騎手コード"] = df_sed["騎手コード"].astype(int)
df_sed["調教師コード"] = df_sed["調教師コード"].astype(int)

df_sed["馬体重"] = pd.to_numeric(df_sed["馬体重"], errors="coerce")
average_value = df_sed["馬体重"].mean()
df_sed["馬体重"] = df_sed["馬体重"].fillna(average_value).astype(float)

df_sed["馬体重増減"] = pd.to_numeric(
    df_sed["馬体重増減"].apply(format_weight_change), errors="coerce"
)
average_value = df_sed["馬体重増減"].mean()
df_sed["馬体重増減"] = df_sed["馬体重増減"].fillna(average_value).astype(float)

df_sed["天候コード"] = df_sed["天候コード"].fillna("0").astype("category")
df_sed["コース"] = df_sed["コース"].fillna("0").astype("category")
df_sed["レース脚質"] = df_sed["レース脚質"].fillna("0").astype("category")
df_sed["本賞金"] = (
    pd.to_numeric(df_sed["本賞金"], errors="coerce").fillna(0).astype(int)
)
df_sed["収得賞金"] = (
    pd.to_numeric(df_sed["収得賞金"], errors="coerce").fillna(0).astype(int)
)
df_sed["レースペース流れ"] = df_sed["レースペース流れ"].fillna("00").astype("category")
df_sed["馬ペース流れ"] = df_sed["馬ペース流れ"].fillna("00").astype("category")
df_sed["4角コース取り"] = df_sed["4角コース取り"].fillna("0").astype("category")

df_sed = df_sed[
    [
        "レースキー_場コード",
        "レースキー_年",
        "レースキー_回",
        "レースキー_日",
        "レースキー_R",
        "馬番",
        "競走成績キー",
        "レース条件_馬場状態",
        "馬成績_着順",
        "馬成績_異常区分",
        "馬成績_タイム",
        "馬成績_斤量",
        "馬成績_確定単勝オッズ",
        "馬成績_確定単勝オッズ_log1p",
        "馬成績_確定単勝人気順位",
        "JRDBデータ_IDM",
        "JRDBデータ_素点",
        "JRDBデータ_馬場差",
        "JRDBデータ_コース取り",
        "JRDBデータ_上昇度コード",
        "JRDBデータ_クラスコード",
        "JRDBデータ_馬体コード",
        "JRDBデータ_気配コード",
        "JRDBデータ_レースペース",
        "JRDBデータ_馬ペース",
        "JRDBデータ_テン指数_boxcox",
        "JRDBデータ_上がり指数_boxcox",
        "JRDBデータ_ペース指数_boxcox",
        "JRDBデータ_レースP指数_boxcox",
        "JRDBデータ_1(2)着タイム差",
        "JRDBデータ_前3Fタイム",
        "JRDBデータ_後3Fタイム",
        "確定複勝オッズ下",
        "確定複勝オッズ下_log1p",
        "10時単勝オッズ_log1p",
        "10時複勝オッズ_log1p",
        "コーナー順位1",
        "コーナー順位2",
        "コーナー順位3",
        "コーナー順位4",
        "前3F先頭差",
        "後3F先頭差",
        "騎手コード",
        "調教師コード",
        "馬体重",
        "馬体重増減",
        "天候コード",
        "コース",
        "レース脚質",
        "本賞金",
        "収得賞金",
        "レースペース流れ",
        "馬ペース流れ",
        "4角コース取り",
    ]
]

save_df_file(str(file_directory), df_sed, "SED")
summary_sed = summarize_object_columns(df_sed)


df_kza = read_df_file(file_directory, "KZA")
df_ksa = read_df_file(file_directory, "KSA")
df_ks = pd.concat([df_ksa, df_kza])
df_ks = df_ks.drop_duplicates().reset_index(drop=True)

df_ks["騎手コード"] = df_ks["騎手コード"].astype(int)
df_ks["所属コード"] = df_ks["所属コード"].astype("category")

df_ks["騎手生年"] = df_ks["生年月日"].str[:4]
df_ks["騎手生年"] = pd.to_numeric(df_ks["騎手生年"], errors="coerce")
average_value = df_ks["騎手生年"].mean()
df_ks["騎手生年"] = df_ks["騎手生年"].fillna(average_value).astype(int)

df_ks["初免許年"] = df_ks["初免許年"].astype(int)
df_ks["見習い区分"] = df_ks["見習い区分"].astype("category")

df_ks["本年平地成績_1着率"] = df_ks["本年平地成績"].apply(
    lambda x: parse_and_calculate_rates(x, 3)[0]
)
average_value = df_ks["本年平地成績_1着率"].mean()
df_ks["本年平地成績_1着率"] = df_ks["本年平地成績_1着率"].fillna(average_value)

df_ks["本年平地成績_2着以内率"] = df_ks["本年平地成績"].apply(
    lambda x: parse_and_calculate_rates(x, 3)[1]
)
average_value = df_ks["本年平地成績_2着以内率"].mean()
df_ks["本年平地成績_2着以内率"] = df_ks["本年平地成績_2着以内率"].fillna(average_value)

df_ks["本年平地成績_3着以内率"] = df_ks["本年平地成績"].apply(
    lambda x: parse_and_calculate_rates(x, 3)[2]
)
average_value = df_ks["本年平地成績_3着以内率"].mean()
df_ks["本年平地成績_3着以内率"] = df_ks["本年平地成績_3着以内率"].fillna(average_value)


df_ks["本年障害成績_1着率"] = df_ks["本年障害成績"].apply(
    lambda x: parse_and_calculate_rates(x, 3)[0]
)
average_value = df_ks["本年障害成績_1着率"].mean()
df_ks["本年障害成績_1着率"] = df_ks["本年障害成績_1着率"].fillna(average_value)

df_ks["本年障害成績_2着以内率"] = df_ks["本年障害成績"].apply(
    lambda x: parse_and_calculate_rates(x, 3)[1]
)
average_value = df_ks["本年障害成績_2着以内率"].mean()
df_ks["本年障害成績_2着以内率"] = df_ks["本年障害成績_2着以内率"].fillna(average_value)

df_ks["本年障害成績_3着以内率"] = df_ks["本年障害成績"].apply(
    lambda x: parse_and_calculate_rates(x, 3)[2]
)
average_value = df_ks["本年障害成績_3着以内率"].mean()
df_ks["本年障害成績_3着以内率"] = df_ks["本年障害成績_3着以内率"].fillna(average_value)

df_ks["昨年平地成績_1着率"] = df_ks["昨年平地成績"].apply(
    lambda x: parse_and_calculate_rates(x, 3)[0]
)
average_value = df_ks["昨年平地成績_1着率"].mean()
df_ks["昨年平地成績_1着率"] = df_ks["昨年平地成績_1着率"].fillna(average_value)

df_ks["昨年平地成績_2着以内率"] = df_ks["昨年平地成績"].apply(
    lambda x: parse_and_calculate_rates(x, 3)[1]
)
average_value = df_ks["昨年平地成績_2着以内率"].mean()
df_ks["昨年平地成績_2着以内率"] = df_ks["昨年平地成績_2着以内率"].fillna(average_value)

df_ks["昨年平地成績_3着以内率"] = df_ks["昨年平地成績"].apply(
    lambda x: parse_and_calculate_rates(x, 3)[2]
)
average_value = df_ks["昨年平地成績_3着以内率"].mean()
df_ks["昨年平地成績_3着以内率"] = df_ks["昨年平地成績_3着以内率"].fillna(average_value)

df_ks["昨年障害成績_1着率"] = df_ks["昨年障害成績"].apply(
    lambda x: parse_and_calculate_rates(x, 3)[0]
)
average_value = df_ks["昨年障害成績_1着率"].mean()
df_ks["昨年障害成績_1着率"] = df_ks["昨年障害成績_1着率"].fillna(average_value)

df_ks["昨年障害成績_2着以内率"] = df_ks["昨年障害成績"].apply(
    lambda x: parse_and_calculate_rates(x, 3)[1]
)
average_value = df_ks["昨年障害成績_2着以内率"].mean()
df_ks["昨年障害成績_2着以内率"] = df_ks["昨年障害成績_2着以内率"].fillna(average_value)

df_ks["昨年障害成績_3着以内率"] = df_ks["昨年障害成績"].apply(
    lambda x: parse_and_calculate_rates(x, 3)[2]
)
average_value = df_ks["昨年障害成績_3着以内率"].mean()
df_ks["昨年障害成績_3着以内率"] = df_ks["昨年障害成績_3着以内率"].fillna(average_value)

df_ks["通算平地成績_1着率"] = df_ks["通算平地成績"].apply(
    lambda x: parse_and_calculate_rates(x, 5)[0]
)
df_ks["通算平地成績_2着以内率"] = df_ks["通算平地成績"].apply(
    lambda x: parse_and_calculate_rates(x, 5)[1]
)
df_ks["通算平地成績_3着以内率"] = df_ks["通算平地成績"].apply(
    lambda x: parse_and_calculate_rates(x, 5)[2]
)

df_ks["通算障害成績_1着率"] = df_ks["通算障害成績"].apply(
    lambda x: parse_and_calculate_rates(x, 5)[0]
)
df_ks["通算障害成績_2着以内率"] = df_ks["通算障害成績"].apply(
    lambda x: parse_and_calculate_rates(x, 5)[1]
)
df_ks["通算障害成績_3着以内率"] = df_ks["通算障害成績"].apply(
    lambda x: parse_and_calculate_rates(x, 5)[2]
)

df_ks = df_ks[
    [
        "騎手コード",
        "所属コード",
        "騎手生年",
        "初免許年",
        "見習い区分",
        "本年平地成績_1着率",
        "本年平地成績_2着以内率",
        "本年平地成績_3着以内率",
        "本年障害成績_1着率",
        "本年障害成績_2着以内率",
        "本年障害成績_3着以内率",
        "昨年平地成績_1着率",
        "昨年平地成績_2着以内率",
        "昨年平地成績_3着以内率",
        "昨年障害成績_1着率",
        "昨年障害成績_2着以内率",
        "昨年障害成績_3着以内率",
        "通算平地成績_1着率",
        "通算平地成績_2着以内率",
        "通算平地成績_3着以内率",
        "通算障害成績_1着率",
        "通算障害成績_2着以内率",
        "通算障害成績_3着以内率",
        "データ年月日",
    ]
]

save_df_file(str(file_directory), df_ks, "KS")
summary_ks = summarize_object_columns(df_ks)


df_ukc = read_df_file(file_directory, "UKC")

df_ukc["血統登録番号"] = df_ukc["血統登録番号"].fillna(0).astype(int)
df_ukc["性別コード"] = df_ukc["性別コード"].astype("category")
df_ukc["毛色コード"] = df_ukc["毛色コード"].fillna("00").astype("category")
df_ukc["馬記号コード"] = df_ukc["馬記号コード"].fillna("").astype("category")
df_ukc["血統情報_父馬名"] = df_ukc["血統情報_父馬名"].fillna("").astype("category")
df_ukc["血統情報_母馬名"] = df_ukc["血統情報_母馬名"].fillna("").astype("category")
df_ukc["血統情報_母父馬名"] = df_ukc["血統情報_母父馬名"].fillna("").astype("category")
df_ukc["馬生年月日"] = df_ukc["生年月日"].astype(int)

df_ukc["父馬生年"] = pd.to_numeric(df_ukc["父馬生年"], errors="coerce")
average_value = df_ukc["父馬生年"].mean()
df_ukc["父馬生年"] = df_ukc["父馬生年"].fillna(average_value).astype(int)

df_ukc["母馬生年"] = pd.to_numeric(df_ukc["母馬生年"], errors="coerce")
average_value = df_ukc["母馬生年"].mean()
df_ukc["母馬生年"] = df_ukc["母馬生年"].fillna(average_value).astype(int)

df_ukc["母父馬生年"] = pd.to_numeric(df_ukc["母父馬生年"], errors="coerce")
average_value = df_ukc["母父馬生年"].mean()
df_ukc["母父馬生年"] = df_ukc["母父馬生年"].fillna(average_value).astype(int)

df_ukc["馬主名"] = df_ukc["馬主名"].fillna("").astype("category")
df_ukc["馬主会コード"] = df_ukc["馬主会コード"].fillna("").astype("category")
df_ukc["生産者名"] = df_ukc["生産者名"].fillna("").astype("category")
df_ukc["産地名"] = df_ukc["産地名"].fillna("").astype("category")
df_ukc["父系統コード"] = df_ukc["父系統コード"].fillna("").astype("category")
df_ukc["母父系統コード"] = df_ukc["母父系統コード"].fillna("").astype("category")
df_ukc["データ年月日"] = df_ukc["データ年月日"].astype(int)

df_ukc = df_ukc[
    [
        "血統登録番号",
        "性別コード",
        "毛色コード",
        "馬記号コード",
        "血統情報_父馬名",
        "血統情報_母馬名",
        "血統情報_母父馬名",
        "馬生年月日",
        "父馬生年",
        "母馬生年",
        "母父馬生年",
        "馬主名",
        "馬主会コード",
        "生産者名",
        "産地名",
        "父系統コード",
        "母父系統コード",
        "データ年月日",
    ]
]

save_df_file(str(file_directory), df_ukc, "UKC")
summary_ukc = summarize_object_columns(df_ukc)


# df_cza = read_df_file(file_directory, "CZA")
# df_csa = read_df_file(file_directory, "CSA")
# df_cha = read_df_file(file_directory, "CHA")
# df_cyb = read_df_file(file_directory, "CYB")
# df_joa = read_df_file(file_directory, "JOA")
# df_kab = read_df_file(file_directory, "KAB")
# df_kka = read_df_file(file_directory, "KKA")
# df_zed = read_df_file(file_directory, "ZED")
# df_zkb = read_df_file(file_directory, "ZKB")
# df_srb = read_df_file(file_directory, "SRB")
# df_skb = read_df_file(file_directory, "SKB")
