import os

# import re
# from datetime import datetime
from dotenv import load_dotenv

# from datetime import datetime

# from tqdm import tqdm

project_path = "../../"
env_file = os.getenv("ENV_FILE", os.path.join(project_path, ".env"))
load_dotenv(env_file)
jrdb_zip_dir = os.getenv("JRDB_ZIP_DIR")
jrdb_txt_dir = os.getenv("JRDB_TXT_DIR")


def get_directories(target_dir):
    # target_dir 内のディレクトリ一覧を取得
    directories = [
        d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))
    ]
    return directories


# 使用例
directories = get_directories(jrdb_zip_dir)
print(directories)
