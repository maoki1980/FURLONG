import pandas as pd
import yaml


# dfの要約を求める関数
def summarize_object_columns(df):
    # データ型の確認
    data_types = df.dtypes

    # 欠損値の数と欠損率
    missing_values_sum = df.isnull().sum()
    missing_values_percentage = (missing_values_sum / len(df)) * 100

    # ユニーク値の数
    unique_values = df.nunique()

    # 結果をまとめたデータフレームの作成
    summary_df = pd.DataFrame(
        {
            "DataType": data_types,
            "MissingVals": missing_values_sum,
            "Missing%": missing_values_percentage,
            "UniqueVals": unique_values,
        }
    )

    return summary_df


# 設定ファイルを読み込む関数
def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config
