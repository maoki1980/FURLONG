import os

import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib_fontja
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


def hex_to_decimal(hex_str):
    return str(int(hex_str, 16))


def fullwidth_to_halfwidth(s):
    if isinstance(s, str):
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


def read_data(input_dir):
    df_all = pd.read_feather(os.path.join(input_dir, "ALL_DATA.feather"))
    df_target = pd.read_feather(os.path.join(input_dir, "DF_TARGET.feather"))
    df_features = pd.read_feather(os.path.join(input_dir, "DF_FEATURES.feather"))
    df_test_features = pd.read_feather(
        os.path.join(input_dir, "DF_TEST_FEATURES.feather")
    )
    return df_all, df_target, df_features, df_test_features


def prepare_data(df_features, df_target, key_columns, group_col, target_col):
    del_cols = key_columns.copy()
    del_cols.remove(group_col)
    X = df_features.drop(del_cols, axis=1)
    y = df_target[target_col]
    return X, y


def map_score(y_true, y_pred):
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    sorted_indices = np.argsort(y_pred)[::-1]
    y_true = np.asarray(y_true)[sorted_indices]
    precisions = np.cumsum(y_true) / (np.arange(len(y_true)) + 1)
    return np.sum(precisions * y_true) / np.sum(y_true)


def precision_at_k(y_true, y_pred, k):
    df = pd.DataFrame({"true": y_true, "pred": y_pred})
    df = df.sort_values("pred", ascending=False).head(k)
    return df["true"].mean()


def mrr_score(y_true, y_pred):
    df = pd.DataFrame({"true": y_true, "pred": y_pred})
    df = df.sort_values("pred", ascending=False)
    df["rank"] = np.arange(1, len(df) + 1)
    df["rr"] = df["true"] / df["rank"]
    return df["rr"].sum() / df["true"].sum()


def train_and_evaluate_model(
    X, y, group_col, params, num_iterations, early_stopping_round, skf
):
    (
        accuracies,
        roc_aucs,
        ndcgs,
        maps,
        precision_at_3s,
        mrrs,
        best_iterations,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    fold_importance_df = pd.DataFrame()

    for fold, (train_index, valid_index) in enumerate(
        tqdm(skf.split(X, y), total=skf.get_n_splits(), desc="Cross-Validation")
    ):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        # クエリーデータの作成
        train_query = X_train.groupby(group_col).size().values.tolist()
        valid_query = X_valid.groupby(group_col).size().values.tolist()
        # group_colを削除
        X_train = X_train.drop(group_col, axis=1)
        X_valid = X_valid.drop(group_col, axis=1)

        train_data = lgb.Dataset(X_train, label=y_train, group=train_query)
        valid_data = lgb.Dataset(
            X_valid, label=y_valid, group=valid_query, reference=train_data
        )

        evals_result = {}
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
                lgb.record_evaluation(eval_result=evals_result),
            ],
        )

        best_iterations.append(model.best_iteration)

        y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
        y_pred_binary = np.where(y_pred > 15, 1, 0)
        y_valid_binary = np.where(y_valid > 15, 1, 0)

        accuracies.append(accuracy_score(y_valid_binary, y_pred_binary))
        roc_aucs.append(roc_auc_score(y_valid_binary, y_pred))
        ndcgs.append(model.best_score["valid_0"]["ndcg@3"])
        maps.append(map_score(y_valid, y_pred))
        precision_at_3s.append(precision_at_k(y_valid, y_pred, 3))
        mrrs.append(mrr_score(y_valid, y_pred))

        fold_importance_df[f"fold_{fold + 1}"] = model.feature_importance()

    features = X.drop(group_col, axis=1).columns
    fold_importance_df.index = features

    return (
        accuracies,
        roc_aucs,
        ndcgs,
        maps,
        precision_at_3s,
        mrrs,
        best_iterations,
        fold_importance_df,
    )


def calculate_metrics(
    accuracies,
    roc_aucs,
    ndcgs,
    maps,
    precision_at_3s,
    mrrs,
    best_iterations,
):
    metrics = {
        "accuracy": (np.mean(accuracies), np.std(accuracies)),
        "roc_auc": (np.mean(roc_aucs), np.std(roc_aucs)),
        "ndcg": (np.mean(ndcgs), np.std(ndcgs)),
        "map": (np.mean(maps), np.std(maps)),
        "precision_at_3": (np.mean(precision_at_3s), np.std(precision_at_3s)),
        "mrr": (np.mean(mrrs), np.std(mrrs)),
        "best_iteration": (np.mean(best_iterations), np.std(best_iterations)),
    }
    return metrics


def display_metrics(metrics):
    for metric, (mean, std) in metrics.items():
        print(f"CV {metric.replace('_', ' ').title()}: {mean:.4f} ± {std:.4f}")


def plot_feature_importance(feature_importance_df, ranking, output_dir):
    feature_importance_df = (
        feature_importance_df.mean(axis=1).sort_values(ascending=False).reset_index()
    )
    feature_importance_df.columns = ["Feature", "Importance"]
    feature_importance_df.to_excel(
        os.path.join(output_dir, "feature_importance.xlsx"), index=False
    )
    feature_importance_df = feature_importance_df.head(ranking)

    matplotlib_fontja.japanize()
    plt.figure(figsize=(10, 10))
    sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
    plt.title(f"Feature Importance Top {ranking}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.savefig(
        os.path.join(output_dir, f"feature_importance_top_{ranking}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def create_final_model(X, y, group_col, params, best_iterations):
    train_query = X.groupby(group_col).size().values.tolist()
    X = X.drop(group_col, axis=1)
    final_train_data = lgb.Dataset(X, label=y, group=train_query)
    final_num_iteration = (
        int(np.max(best_iterations)) if int(np.max(best_iterations)) > 100 else 100
    )
    final_model = lgb.train(
        params=params, train_set=final_train_data, num_boost_round=final_num_iteration
    )
    return final_model


def prepare_test_data(df_test_features, key_columns, final_model):
    X_test = df_test_features.drop(key_columns, axis=1)
    y_test_pred = final_model.predict(X_test, num_iteration=final_model.best_iteration)
    return y_test_pred


def merge_results(df_test_features, y_test_pred, df_all, target_col):
    df_test_features_with_predictions = df_test_features.copy()
    df_test_features_with_predictions[f"{target_col}予想"] = y_test_pred
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
    return df_result


def transform_columns(df):
    df = df.astype(str)

    for col in df.columns:
        df[col] = df[col].apply(fullwidth_to_halfwidth)

    df["IN_レースキー_回"] = df["IN_レースキー_回"] + "回"

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

    df["IN_レースキー_日"] = df["IN_レースキー_日"].apply(hex_to_decimal) + "日目"
    df["IN_レースキー_R"] = df["IN_レースキー_R"] + "R"

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

    race_name = (
        df["IN_回数"]
        + df["IN_回数"].apply(lambda x: " " if x else "")
        + df["IN_レース名"]
    )
    df = pd.concat([df, race_name.rename("レース名")], axis=1)

    type_dict = {
        "0": "",
        "1": "芝",
        "2": "ダート",
        "3": "障害",
    }
    df["IN_レース条件_芝ダ障害コード"] = df["IN_レース条件_芝ダ障害コード"].map(
        type_dict
    )

    df["IN_レース条件_距離"] = df["IN_レース条件_距離"] + "m"
    df["レースキー"] = (
        df["IN_レースキー_回"] + df["IN_レースキー_場コード"] + df["IN_レースキー_日"]
    )

    df["発走日時"] = pd.to_datetime(
        df["IN_年月日"] + df["IN_発走時間"], format="%Y%m%d%H%M"
    )

    df["IN_馬番"] = df["IN_馬番"].astype(int)
    df["IN_枠番"] = df["IN_枠番"].astype(int)
    df[f"{target_col}予想"] = df[f"{target_col}予想"].astype(float)

    df = df.sort_values(by=["IN_レースキー_場コード", "IN_レースキー_R", "IN_馬番"])

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
            f"{target_col}予想",
        ]
    ]

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
        f"{target_col}予想",
    ]
    df.columns = column_names

    return df


# シードの設定
RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)

# 環境変数の読み込み
project_path = "../../"
env_file = os.getenv("ENV_FILE", os.path.join(project_path, ".env"))
load_dotenv(env_file)
file_directory = os.getenv("DF_DIR")
pred_directory = os.path.join(str(os.getenv("PRED_DIR")), "RANKING")

# 結果出力ディレクトリ作成
os.makedirs(pred_directory, exist_ok=True)

# データの読み込み
df_all, df_target, df_features, df_test_features = read_data(file_directory)

# 目的変数を選択
target_col = "着順カテゴリ"
key_columns = [
    "IN_レースキー",
    "IN_レースキー_場コード",
    "IN_レースキー_回",
    "IN_レースキー_日",
    "IN_レースキー_R",
    "_merge",
]
group_col = "IN_レースキー"

# 説明変数と目的変数の分割
X, y = prepare_data(df_features, df_target, key_columns, group_col, target_col)

# モデルパラメーター
params = {
    "objective": "lambdarank",
    "metric": "ndcg",
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
(
    accuracies,
    roc_aucs,
    ndcgs,
    maps,
    precision_at_3s,
    mrrs,
    best_iterations,
    fold_importance_df,
) = train_and_evaluate_model(
    X, y, group_col, params, num_iterations, early_stopping_round, skf
)


# 評価結果の表示
metrics = calculate_metrics(
    accuracies,
    roc_aucs,
    ndcgs,
    maps,
    precision_at_3s,
    mrrs,
    best_iterations,
)
display_metrics(metrics)

# 特徴量の重要度の可視化
plot_feature_importance(fold_importance_df, 50, pred_directory)

# 全データを用いて最終モデルを再学習する
final_model = create_final_model(X, y, group_col, params, best_iterations)

# テストデータの予測
y_test_pred = prepare_test_data(df_test_features, key_columns, final_model)

# 結果のマージ
df_result = merge_results(df_test_features, y_test_pred, df_all, target_col)

# カラムの変換
df_transformed = transform_columns(df_result)

# 結果の保存
df_transformed.to_excel(os.path.join(pred_directory, "result.xlsx"), index=False)
