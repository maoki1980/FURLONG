import json
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm


# テキストファイルを指定されたフォーマットに変換してデータフレームとして保存する関数
def format_and_save_data(file_prefix, input_directory, output_directory, spec_filepath):
    # JSTを設定
    tokyo_tz = timezone(timedelta(hours=9))
    # 開始時刻を取得
    start_time = datetime.now(tokyo_tz)
    print(f"Start time (Tokyo): {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    # ファイルリストを取得
    txt_files = get_files_with_prefix(input_directory, file_prefix)
    # ファイルを結合
    all_records = []
    for txt_file in tqdm(txt_files, desc=f"Processing {file_prefix} files"):
        formatted_records = format_file_data(txt_file, spec_filepath)
        all_records.extend(formatted_records)
    # データフレームに変換してファイルに保存
    df = pd.DataFrame(all_records)
    save_dataframe(df, output_directory, file_prefix)
    # 終了時刻を取得
    end_time = datetime.now(tokyo_tz)
    print(f"End time (Tokyo): {end_time.strftime('%Y-%m-%d %H:%M:%S')}")


# データフレームをfeather形式で保存する関数
def save_dataframe(df, output_directory, file_prefix):
    file_path = os.path.join(output_directory, f"{file_prefix}.feather")
    df.to_feather(file_path)


# 指定したディレクトリ内の指定したプレフィックスで始まるファイルのリストを取得する関数
def get_files_with_prefix(directory, prefix):
    all_files = os.listdir(directory)
    matching_files = [f for f in all_files if f.startswith(prefix)]
    matching_file_paths = [os.path.join(directory, f) for f in matching_files]

    return matching_file_paths


# テキストファイルのデータを指定のフォーマットに従って整形する関数
def format_file_data(data_filepath, spec_filepath):
    with open(spec_filepath, "r", encoding="cp932") as json_file:
        column_specs = json.load(json_file)

    with open(data_filepath, "r", encoding="cp932") as txt_file:
        data = txt_file.read()

    lines = data.strip().split("\n")
    records = []

    for line in lines:
        formatted_record = {}
        line_bytes = line.encode("cp932", errors="ignore")

        for column_name, byte_range in column_specs["データ位置"].items():
            start_byte = byte_range["start_byte"]
            end_byte = byte_range["end_byte"]
            column_value = line_bytes[start_byte : end_byte + 1]
            formatted_record[column_name] = column_value.decode(
                "cp932", errors="ignore"
            ).strip()

        records.append(formatted_record)
    return records


# 環境変数の読み込み
project_path = "../../"
env_file = os.getenv("ENV_FILE", os.path.join(project_path, ".env"))
load_dotenv(env_file)

jrdb_spec_directory = os.getenv("JRDB_SPEC_JSON_DIR")
jrdb_txt_directory = os.getenv("JRDB_TXT_DIR")
dataframe_output_directory = os.getenv("DF_DIR")

# CZAファイル群を処理
format_and_save_data(
    file_prefix="CZA",
    input_directory=os.path.join(str(jrdb_txt_directory), "Cs"),
    output_directory=dataframe_output_directory,
    spec_filepath=os.path.join(str(jrdb_spec_directory), "CSA.json"),
)

# CSAファイル群を処理
format_and_save_data(
    file_prefix="CSA",
    input_directory=os.path.join(str(jrdb_txt_directory), "Cs"),
    output_directory=dataframe_output_directory,
    spec_filepath=os.path.join(str(jrdb_spec_directory), "CSA.json"),
)

# HJCファイル群を処理
format_and_save_data(
    file_prefix="HJC",
    input_directory=os.path.join(str(jrdb_txt_directory), "Hjc"),
    output_directory=dataframe_output_directory,
    spec_filepath=os.path.join(str(jrdb_spec_directory), "HJC.json"),
)

# KZAファイル群を処理
format_and_save_data(
    file_prefix="KZA",
    input_directory=os.path.join(str(jrdb_txt_directory), "Ks"),
    output_directory=dataframe_output_directory,
    spec_filepath=os.path.join(str(jrdb_spec_directory), "KSA.json"),
)

# KSAファイル群を処理
format_and_save_data(
    file_prefix="KSA",
    input_directory=os.path.join(str(jrdb_txt_directory), "Ks"),
    output_directory=dataframe_output_directory,
    spec_filepath=os.path.join(str(jrdb_spec_directory), "KSA.json"),
)

# BACファイル群を処理
format_and_save_data(
    file_prefix="BAC",
    input_directory=os.path.join(str(jrdb_txt_directory), "Paci"),
    output_directory=dataframe_output_directory,
    spec_filepath=os.path.join(str(jrdb_spec_directory), "BAC.json"),
)

# CHAファイル群を処理
format_and_save_data(
    file_prefix="CHA",
    input_directory=os.path.join(str(jrdb_txt_directory), "Paci"),
    output_directory=dataframe_output_directory,
    spec_filepath=os.path.join(str(jrdb_spec_directory), "CHA.json"),
)

# CYBファイル群を処理
format_and_save_data(
    file_prefix="CYB",
    input_directory=os.path.join(str(jrdb_txt_directory), "Paci"),
    output_directory=dataframe_output_directory,
    spec_filepath=os.path.join(str(jrdb_spec_directory), "CYB.json"),
)

# JOAファイル群を処理
format_and_save_data(
    file_prefix="JOA",
    input_directory=os.path.join(str(jrdb_txt_directory), "Paci"),
    output_directory=dataframe_output_directory,
    spec_filepath=os.path.join(str(jrdb_spec_directory), "JOA.json"),
)

# KABファイル群を処理
format_and_save_data(
    file_prefix="KAB",
    input_directory=os.path.join(str(jrdb_txt_directory), "Paci"),
    output_directory=dataframe_output_directory,
    spec_filepath=os.path.join(str(jrdb_spec_directory), "KAB.json"),
)

# KKAファイル群を処理
format_and_save_data(
    file_prefix="KKA",
    input_directory=os.path.join(str(jrdb_txt_directory), "Paci"),
    output_directory=dataframe_output_directory,
    spec_filepath=os.path.join(str(jrdb_spec_directory), "KKA.json"),
)

# KYIファイル群を処理
format_and_save_data(
    file_prefix="KYI",
    input_directory=os.path.join(str(jrdb_txt_directory), "Paci"),
    output_directory=dataframe_output_directory,
    spec_filepath=os.path.join(str(jrdb_spec_directory), "KYI.json"),
)

# UKCファイル群を処理
format_and_save_data(
    file_prefix="UKC",
    input_directory=os.path.join(str(jrdb_txt_directory), "Paci"),
    output_directory=dataframe_output_directory,
    spec_filepath=os.path.join(str(jrdb_spec_directory), "UKC.json"),
)

# ZEDファイル群を処理
format_and_save_data(
    file_prefix="ZED",
    input_directory=os.path.join(str(jrdb_txt_directory), "Paci"),
    output_directory=dataframe_output_directory,
    spec_filepath=os.path.join(str(jrdb_spec_directory), "ZED.json"),
)

# ZKBファイル群を処理
format_and_save_data(
    file_prefix="ZKB",
    input_directory=os.path.join(str(jrdb_txt_directory), "Paci"),
    output_directory=dataframe_output_directory,
    spec_filepath=os.path.join(str(jrdb_spec_directory), "ZKB.json"),
)

# SEDファイル群を処理
format_and_save_data(
    file_prefix="SED",
    input_directory=os.path.join(str(jrdb_txt_directory), "Sed"),
    output_directory=dataframe_output_directory,
    spec_filepath=os.path.join(str(jrdb_spec_directory), "SED.json"),
)

# SRBファイル群を処理
format_and_save_data(
    file_prefix="SRB",
    input_directory=os.path.join(str(jrdb_txt_directory), "Sed"),
    output_directory=dataframe_output_directory,
    spec_filepath=os.path.join(str(jrdb_spec_directory), "SRB.json"),
)

# SKBファイル群を処理
format_and_save_data(
    file_prefix="SKB",
    input_directory=os.path.join(str(jrdb_txt_directory), "Skb"),
    output_directory=dataframe_output_directory,
    spec_filepath=os.path.join(str(jrdb_spec_directory), "SKB.json"),
)
