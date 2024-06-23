import json
import os
import re

from dotenv import load_dotenv


def read_txt_file(file_path):
    with open(file_path, "r", encoding="cp932", errors="ignore") as file:
        return file.readlines()


# キー名を整形する関数
def format_key_name(key_name):
    # 全角英数文字を半角に変換
    key_name = re.sub(
        r"[Ａ-Ｚａ-ｚ０-９]", lambda x: chr(ord(x.group(0)) - 0xFEE0), key_name
    )

    # キー名の最後にアンダーバーがついている場合はそれを削除
    if key_name.endswith("_"):
        key_name = key_name[:-1]

    return key_name


# 固定長テキストデータの仕様を解析する関数(OCCがある場合)
def parse_spec_occ(lines):
    data_info = {}
    current_parent = None
    current_children = []
    cumulative_byte_offset = 0  # 累積バイトオフセットを追跡する変数

    for line in lines:
        # バイト列にエンコード
        line_bytes = line.encode("cp932", errors="ignore")

        # インデントチェック
        indent_match = re.match(r"^\s*", line)
        indent_level = len(indent_match.group()) if indent_match else 0
        # 項目名 0桁目から, OCC 16桁目から, BYTE 24桁目から, TYPE 32桁目から, 相対 40桁目から, 備考 48桁目から
        match = re.match(rb"(.{16})(.{8})(.{8})(.{8})(.{8})(.*)", line_bytes)

        if match:
            item_name = match.group(1).decode("cp932", errors="ignore").strip()
            occ = match.group(2).decode("cp932", errors="ignore").strip()
            byte_len = match.group(3).decode("cp932", errors="ignore").strip()
            data_type = match.group(4).decode("cp932", errors="ignore").strip()
            start_byte = match.group(5).decode("cp932", errors="ignore").strip()
            note = (
                match.group(6).decode("cp932", errors="ignore").strip()
                if match.group(6)
                else ""
            )

            try:
                occ = int(occ) if occ else 1
                byte_len = int(byte_len) if byte_len else 0
                start_byte = int(start_byte) - 1 if start_byte else 0
                end_byte = start_byte + byte_len - 1
            except ValueError:
                continue

            # キー名を整形
            item_name = format_key_name(item_name)
            if item_name is None:
                continue

            if indent_level == 0 and line.strip():
                # 既存の親キーとその子キーをdata_infoに追加
                if current_parent:
                    for i in range(1, current_parent["occ"] + 1):
                        for child in current_children:
                            if current_parent["occ"] == 1:
                                nested_key = f"{current_parent['name']}_{child['name']}"
                            else:
                                nested_key = (
                                    f"{current_parent['name']}{i}_{child['name']}"
                                )
                            nested_key = format_key_name(nested_key)
                            if nested_key:
                                data_info[nested_key] = {
                                    "start_byte": cumulative_byte_offset,
                                    "end_byte": cumulative_byte_offset
                                    + child["byte_length"]
                                    - 1,
                                    "byte_length": child["byte_length"],
                                    "type": child["type"],
                                    "note": child["note"],
                                }
                                cumulative_byte_offset += child["byte_length"]
                    # 子キーがない場合、親キーのみを追加
                    if not current_children:
                        parent_key = format_key_name(current_parent["name"])
                        if parent_key:
                            data_info[parent_key] = {
                                "start_byte": cumulative_byte_offset,
                                "end_byte": cumulative_byte_offset
                                + current_parent["byte_len"]
                                - 1,
                                "byte_length": current_parent["byte_len"],
                                "type": current_parent["data_type"],
                                "note": current_parent["note"],
                            }
                            cumulative_byte_offset += current_parent["byte_len"]

                # 新しい親キーとして処理
                current_parent = {
                    "name": item_name,
                    "occ": occ,
                    "start_byte": start_byte,
                    "byte_len": byte_len,
                    "data_type": data_type,
                    "note": note,
                }
                current_children = []
            else:
                # 子キーとして処理
                child_name = format_key_name(item_name)
                if child_name:
                    current_children.append(
                        {
                            "name": child_name,
                            "start_byte": start_byte,
                            "end_byte": end_byte,
                            "byte_length": byte_len,
                            "type": data_type,
                            "note": note,
                        }
                    )

    # 最後の親キーとその子キーをdata_infoに追加
    if current_parent:
        for i in range(1, current_parent["occ"] + 1):
            for child in current_children:
                if current_parent["occ"] == 1:
                    nested_key = f"{current_parent['name']}_{child['name']}"
                else:
                    nested_key = f"{current_parent['name']}{i}_{child['name']}"
                nested_key = format_key_name(nested_key)
                if nested_key:
                    data_info[nested_key] = {
                        "start_byte": cumulative_byte_offset,
                        "end_byte": cumulative_byte_offset + child["byte_length"] - 1,
                        "byte_length": child["byte_length"],
                        "type": child["type"],
                        "note": child["note"],
                    }
                    cumulative_byte_offset += child["byte_length"]
        # 子キーがない場合、親キーのみを追加
        if not current_children:
            parent_key = format_key_name(current_parent["name"])
            if parent_key:
                data_info[parent_key] = {
                    "start_byte": cumulative_byte_offset,
                    "end_byte": cumulative_byte_offset + current_parent["byte_len"] - 1,
                    "byte_length": current_parent["byte_len"],
                    "type": current_parent["data_type"],
                    "note": current_parent["note"],
                }
                cumulative_byte_offset += current_parent["byte_len"]

    return data_info

# 固定長テキストデータの仕様を解析する関数(OCCがある場合)
def parse_spec(lines):
    data_info = {}

    for line in lines:
        # バイト列にエンコード
        line_bytes = line.encode("cp932", errors="ignore")

        # インデントチェック
        indent_match = re.match(r"^\s*", line)
        indent_level = len(indent_match.group()) if indent_match else 0
        # 項目名 0桁目から, OCC 16桁目から, BYTE 24桁目から, TYPE 32桁目から, 相対 40桁目から, 備考 48桁目から
        match = re.match(rb"(.{24})(.{8})(.{8})(\d+)\s+(.*)", line_bytes)

        if indent_level == 0 and line.strip() and not match:
            last_non_indented_key = line.strip()
            last_non_indented_key = format_key_name(last_non_indented_key)

        if match:
            item_name = match.group(1).decode("cp932", errors="ignore").strip()
            byte_len = match.group(2).decode("cp932", errors="ignore").strip()
            data_type = match.group(3).decode("cp932", errors="ignore").strip()
            start_byte = match.group(4).decode("cp932", errors="ignore").strip()
            note = (
                match.group(5).decode("cp932", errors="ignore").strip()
                if match.group(5)
                else ""
            )

            try:
                byte_len = int(byte_len) if byte_len else 0
                start_byte = int(start_byte) - 1 if start_byte else 0
                end_byte = start_byte + byte_len - 1
            except ValueError:
                continue

            # キー名を整形
            item_name = format_key_name(item_name)
            if indent_level == 0:
                last_non_indented_key = item_name
                data_info[last_non_indented_key] = {
                    "start_byte": start_byte,
                    "end_byte": end_byte,
                    "byte_length": byte_len,
                    "type": data_type,
                    "note": note,
                }
            else:
                nested_key = f"{last_non_indented_key}_{item_name}"
                data_info[nested_key] = {
                    "start_byte": start_byte,
                    "end_byte": end_byte,
                    "byte_length": byte_len,
                    "type": data_type,
                    "note": note,
                }

    return data_info


# 基本情報を抽出する関数
def extract_basic_info(lines):
    data_name = None
    record_length = None

    for line in lines:
        # データ名を抽出
        if not data_name:
            data_name_match = re.search(r"(\S+データ仕様)", line)
            if data_name_match:
                data_name = data_name_match.group(1)

        # レコード長を抽出
        if not record_length:
            record_length_match = re.search(r"レコード長：(\d+)BYTE", line)
            if record_length_match:
                record_length = int(record_length_match.group(1))

    return data_name, record_length


# バージョン情報を抽出する関数
def extract_version_info(lines):
    for line in reversed(lines):
        if re.match(r"\s*第[\d\uFF10-\uFF19]+版", line):
            return line.strip()


project_path = "../../"
env_file = os.getenv("ENV_FILE", os.path.join(project_path, ".env"))
load_dotenv(env_file)
jrdb_spec_dir = os.getenv("JRDB_SPEC_DIR")
jrdb_info_dir = os.getenv("JRDB_INFO_DIR")

# JRDB_SPEC_DIRディレクトリ内のすべてのtxtファイルを処理
for file_name in os.listdir(jrdb_spec_dir):
    if file_name.endswith(".txt"):
        spec_file_path = os.path.join(str(jrdb_spec_dir), file_name)
        output_json_path = os.path.join(
            str(jrdb_info_dir), file_name.replace(".txt", ".json")
        )

        # テキストファイルを読み込む
        lines = read_txt_file(spec_file_path)

        # 固定長テキストデータの仕様を解析
        if file_name.lower().startswith("hjc"):
            data_info = parse_spec_occ(lines)
        else:
            data_info = parse_spec(lines)

        # 基本情報を抽出
        data_name, record_length = extract_basic_info(lines)

        # バージョン情報を抽出
        version_info = extract_version_info(lines)

        # JSON形式で保存する
        output_json = {
            "基本情報": {
                "data_name": data_name,
                "record_length": record_length,
                "version": version_info,
            },
            "データ位置": data_info,
        }

        # JSONファイルに保存する
        with open(output_json_path, "w", encoding="cp932") as json_file:
            json.dump(output_json, json_file, ensure_ascii=False, indent=4)

        print(f"Processing of {file_name} completed.")
