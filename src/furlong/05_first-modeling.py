import os

import lightgbm as lgb
import pandas as pd
from dotenv import load_dotenv


# 16進数から10進数の数値文字列に変換する関数
def hex_to_decimal(hex_str):
    return str(int(hex_str, 16))


# 全角文字を半角に変換する関数
def fullwidth_to_halfwidth(s):
    if isinstance(s, str):
        # 変換テーブルを使用して全角を半角に変換
        return s.translate(
            str.maketrans(
                "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
                "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"
                "０１２３４５６７８９・．",
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "abcdefghijklmnopqrstuvwxyz"
                "0123456789･.",
            )
        )
    return s


# 環境変数の読み込み
project_path = "../../"
env_file = os.getenv("ENV_FILE", os.path.join(project_path, ".env"))
load_dotenv(env_file)
file_directory = os.getenv("DF_DIR")

# データの読込み
df_all = pd.read_feather(os.path.join(str(file_directory), "ALL_DATA.feather"))
df_target = pd.read_feather(os.path.join(str(file_directory), "DF_TARGET.feather"))
df_features = pd.read_feather(os.path.join(str(file_directory), "DF_FEATURES.feather"))
df_test_features = pd.read_feather(
    os.path.join(str(file_directory), "DF_TEST_FEATURES.feather")
)

# 目的変数を選択
target_col = "順位カテゴリ"
# 学習に使わないカラム
key_columns = [
    "IN_レースキー",
    "IN_レースキー_場コード",
    "IN_レースキー_回",
    "IN_レースキー_日",
    "IN_レースキー_R",
    "_merge",
]
# 説明変数と目的変数の分割
X = df_features.drop(key_columns, axis=1)
y = df_target[target_col]

# モデルパラメーター
params = {
    "objective": "multiclass",  # 多値分類の場合
    "num_class": len(y.unique()),  # クラス数を指定
    "metric": "multi_logloss",  # 多値分類の場合
    "boosting_type": "gbdt",
    "device": "gpu",
    "verbose": -1,
}

# 全データを用いて最終モデルを再学習する
final_train_data = lgb.Dataset(X, label=y)
final_model = lgb.train(
    params=params,
    train_set=final_train_data,
    valid_sets=[final_train_data],
)

# テストデータの予測
y_test_pred = final_model.predict(
    df_test_features.drop(key_columns, axis=1), num_iteration=final_model.best_iteration
)
# 予測結果をクラスに変換
y_test_pred_classes = [list(x).index(max(x)) for x in y_test_pred]
# df_test_featuresとdf_test_predictionsを結合する
df_test_features_with_predictions = df_test_features.copy()
df_test_features_with_predictions["順位カテゴリ予想"] = y_test_pred_classes
df_result = pd.merge(
    df_test_features_with_predictions,
    df_all,
    on=[
        "IN_レースキー",
        "IN_レースキー_場コード",
        "IN_レースキー_回",
        "IN_レースキー_日",
        "IN_レースキー_R",
        "IN_馬番",
        "IN_枠番",
        "IN_レース条件_距離",
        "IN_レース条件_芝ダ障害コード",
        "IN_レース条件_グレード",
    ],
    how="left",
)

df = df_result.copy()
df = df.astype(str)

# 全角英数字記号を半角に変換
for col in df.columns:
    df[col] = df[col].apply(fullwidth_to_halfwidth)

# 回を付ける
df["IN_レースキー_回"] = df["IN_レースキー_回"] + "回"

# 場名を辞書を用いて変換
course_dic = {
    "01": "札幌",
    "02": "函館",
    "03": "福島",
    "04": "新潟",
    "05": "東京",
    "06": "中山",
    "07": "中京",
    "08": "京都",
    "09": "阪神",
    "10": "小倉",
    "21": "旭川",
    "22": "札幌",
    "23": "門別",
    "24": "函館",
    "25": "盛岡",
    "26": "水沢",
    "27": "上山",
    "28": "新潟",
    "29": "三条",
    "30": "足利",
    "31": "宇都",
    "32": "高崎",
    "33": "浦和",
    "34": "船橋",
    "35": "大井",
    "36": "川崎",
    "37": "金沢",
    "38": "笠松",
    "39": "名古",
    "40": "中京",
    "41": "園田",
    "42": "姫路",
    "43": "益田",
    "44": "福山",
    "45": "高知",
    "46": "佐賀",
    "47": "荒尾",
    "48": "中津",
    "61": "英国",
    "62": "アイルランド",
    "63": "仏国",
    "64": "伊国",
    "65": "独国",
    "66": "米国",
    "67": "カナダ",
    "68": "UAE(アラブ首長国連邦)",
    "69": "オーストラリア",
    "70": "ニュージーランド",
    "71": "香港",
    "72": "チリ",
    "73": "シンガポール",
    "74": "スウェーデン",
    "75": "マカ",
    "76": "オーストリア",
    "77": "トルコ",
    "78": "カタール",
    "79": "韓国",
}
df["IN_レースキー_場コード"] = df["IN_レースキー_場コード"].map(course_dic)

# 16進数を10進数に変換
df["IN_レースキー_日"] = df["IN_レースキー_日"].apply(hex_to_decimal) + "日目"

# Rを付ける
df["IN_レースキー_R"] = df["IN_レースキー_R"] + "R"

# ソート
df = df.sort_values(by=["IN_レースキー_場コード", "IN_レースキー_R", "IN_馬番"])

# レースグレードを変換
grade_dict = {
    "0": "",
    "1": "G1レース",
    "2": "G2レース",
    "3": "G3レース",
    "4": "重賞レース",
    "5": "特別レース",
    "6": "リステッド競走",
}
df["IN_レース条件_グレード"] = df["IN_レース条件_グレード"].map(grade_dict)

# 回数とレース名を結合
race_name = (
    df["IN_回数"] + df["IN_回数"].apply(lambda x: " " if x else "") + df["IN_レース名"]
)
df = pd.concat([df, race_name.rename("レース名")], axis=1)

# 芝ダ障害コードを変換
type_dict = {
    "0": "",
    "1": "芝",
    "2": "ダート",
    "3": "障害",
}
df["IN_レース条件_芝ダ障害コード"] = df["IN_レース条件_芝ダ障害コード"].map(type_dict)

# mを付ける
df["IN_レース条件_距離"] = df["IN_レース条件_距離"] + "m"

# レースキー名
df["レースキー"] = (
    df["IN_レースキー_回"] + df["IN_レースキー_場コード"] + df["IN_レースキー_日"]
)

# 発走日時をフォーマット
df["発走日時"] = df["IN_年月日"] + df["IN_発走時間"]
df["発走日時"] = pd.to_datetime(df["発走日時"], format="%Y%m%d%H%M")

# 数値に変換
df["IN_枠番"] = df["IN_枠番"].astype(int)
df["IN_馬番"] = df["IN_馬番"].astype(int)
df["順位カテゴリ予想"] = df["順位カテゴリ予想"].astype(int)

# 列を抽出
df = df[
    [
        "レースキー",
        "IN_レースキー_R",
        "IN_レース条件_グレード",
        "レース名",
        "IN_レース条件_芝ダ障害コード",
        "IN_レース条件_距離",
        "発走日時",
        "IN_枠番",
        "IN_馬番",
        "IN_馬名",
        "IN_騎手名",
        "順位カテゴリ予想",
    ]
]

# 列をリネーム
column_names = [
    "開催情報",
    "レース番号",
    "レースグレード",
    "レース名",
    "コース種類",
    "コース距離",
    "発走日時",
    "枠番",
    "馬番",
    "馬名",
    "騎手名",
    "順位カテゴリ予想",
]
df.columns = column_names

df.to_excel(os.path.join(str(file_directory), "result.xlsx"), index=False)
