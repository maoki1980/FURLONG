import os

import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib_fontja
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


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


# シードの設定
RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)

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
target_col = "複勝"
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
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "device": "gpu",
    "verbose": -1,
    "seed": RANDOM_SEED,
    "bagging_seed": RANDOM_SEED,
    "feature_fraction_seed": RANDOM_SEED,
}
num_iterations = 10000
early_stopping_round = 100

# Stratified K-Fold Cross-Validationの設定
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

# 交差検証の実行
accuracies = []
precisions = []
recalls = []
f1s = []
f2s = []
log_losses = []
best_thresholds = []
best_iterations = []
roc_aucs = []
fold_importance_df = pd.DataFrame()

for fold, (train_index, valid_index) in enumerate(
    tqdm(skf.split(X, y), total=skf.get_n_splits(), desc="Cross-Validation")
):
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    model = lgb.train(
        params=params,
        train_set=train_data,
        valid_sets=[valid_data],
        num_boost_round=num_iterations,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=early_stopping_round,
                first_metric_only=False,
                verbose=False,
            ),
        ],
    )

    # 最適なイテレーション数を記録
    best_iterations.append(model.best_iteration)

    # 予測
    y_pred = model.predict(X_valid, num_iteration=model.best_iteration)

    # ROC曲線をプロットして最適な閾値を見つける
    fpr, tpr, thresholds = roc_curve(y_valid, y_pred)
    roc_auc = auc(fpr, tpr)
    roc_aucs.append(roc_auc)

    # Youden's J statisticを用いて最適な閾値を選択
    j_scores = tpr - fpr
    best_threshold = thresholds[np.argmax(j_scores)]
    best_thresholds.append(best_threshold)

    # 予測値からクラスに変換
    y_pred_classes = [1 if x >= best_threshold else 0 for x in y_pred]

    accuracies.append(accuracy_score(y_valid, y_pred_classes))
    precisions.append(precision_score(y_valid, y_pred_classes, average="binary"))
    recalls.append(recall_score(y_valid, y_pred_classes, average="binary"))
    f1s.append(f1_score(y_valid, y_pred_classes, average="binary"))
    f2s.append(fbeta_score(y_valid, y_pred_classes, beta=2, average="binary"))
    log_losses.append(log_loss(y_valid, y_pred))

    fold_importance_df[f"fold_{fold + 1}"] = model.feature_importance()

# 評価結果の表示
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
mean_precision = np.mean(precisions)
std_precision = np.std(precisions)
mean_recall = np.mean(recalls)
std_recall = np.std(recalls)
mean_f1 = np.mean(f1s)
std_f1 = np.std(f1s)
mean_f2 = np.mean(f2s)
std_f2 = np.std(f2s)
mean_log_loss = np.mean(log_losses)
std_log_loss = np.std(log_losses)
mean_roc_auc = np.mean(roc_aucs)
std_roc_auc = np.std(roc_aucs)
mean_best_threshold = np.mean(best_thresholds)
std_best_threshold = np.std(best_thresholds)
mean_best_iteration = np.mean(best_iterations)
std_best_iteration = np.std(best_iterations)

print(f"CV Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
print(f"CV Precision: {mean_precision:.4f} ± {std_precision:.4f}")
print(f"CV Recall: {mean_recall:.4f} ± {std_recall:.4f}")
print(f"CV F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
print(f"CV F2 Score: {mean_f2:.4f} ± {std_f2:.4f}")
print(f"CV Log Loss: {mean_log_loss:.4f} ± {std_log_loss:.4f}")
print(f"CV ROC AUC: {mean_roc_auc:.4f} ± {std_roc_auc:.4f}")
print(f"CV Best Threshold: {mean_best_threshold:.4f} ± {std_best_threshold:.4f}")
print(f"CV Best Iteration: {mean_best_iteration:.4f} ± {std_best_iteration:.4f}")

# 特徴量の重要度の可視化
ranking = 50
feature_importance_df = (
    fold_importance_df.mean(axis=1).sort_values(ascending=False)
).reset_index(drop=True)

feature_importance_df = pd.DataFrame(
    {"Feature": X.columns, "Importance": feature_importance_df}
).sort_values(by="Importance", ascending=False)

feature_importance_df.reset_index(drop=True).to_excel(
    os.path.join(str(file_directory), "feature_importance.xlsx"), index=False
)
feature_importance_df = feature_importance_df.head(ranking)

feature_importance_df = feature_importance_df.reset_index(drop=True).set_index(
    "Feature"
)

matplotlib_fontja.japanize()
plt.figure(figsize=(10, 10))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
plt.title(f"Feature Importance Top {ranking}")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# 全データを用いて最終モデルを再学習する
final_train_data = lgb.Dataset(X, label=y)
final_num_iteration = (
    int(np.max(best_iterations)) if int(np.max(best_iterations)) > 100 else 100
)
final_model = lgb.train(
    params=params,
    train_set=final_train_data,
    num_boost_round=final_num_iteration,
)

# テストデータの予測
X_test = df_test_features.drop(key_columns, axis=1)
y_test_pred = final_model.predict(X_test, num_iteration=final_model.best_iteration)
# 予測結果をクラスに変換
y_test_pred_classes = [1 if x >= mean_best_threshold else 0 for x in y_test_pred]
# df_test_featuresとdf_test_predictionsを結合する
df_test_features_with_predictions = df_test_features.copy()
df_test_features_with_predictions[f"{target_col}予想"] = y_test_pred_classes
df_test_features_with_predictions[f"{target_col}確率"] = y_test_pred
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
df["IN_馬番"] = df["IN_馬番"].astype(int)
df["IN_枠番"] = df["IN_枠番"].astype(int)
df[f"{target_col}予想"] = df[f"{target_col}予想"].astype(int)
df[f"{target_col}確率"] = df[f"{target_col}確率"].astype(float)

# ソート
df = df.sort_values(by=["IN_レースキー_場コード", "IN_レースキー_R", "IN_馬番"])

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
        f"{target_col}確率",
        f"{target_col}予想",
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
    f"{target_col}確率",
    f"{target_col}予想",
]
df.columns = column_names

df.to_excel(os.path.join(str(file_directory), "result.xlsx"), index=False)
