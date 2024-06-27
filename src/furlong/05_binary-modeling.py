import os

import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib_fontja
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    fbeta_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
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


def prepare_data(df_features, df_target, key_columns, target_col):
    X = df_features.drop(key_columns, axis=1)
    y = df_target[target_col]
    return X, y


def plot_learning_curve(evals_result, metric, fold, output_dir):
    epochs = len(evals_result["valid_0"][metric])
    x_axis = range(0, epochs)

    # DataFrameに変換
    df_learning_curve = pd.DataFrame(
        {
            "epoch": x_axis,
            "validation": evals_result["valid_0"][metric],
        }
    )

    # Excelファイルとして保存
    df_learning_curve.to_excel(
        os.path.join(output_dir, f"learning_curve_fold_{fold + 1}.xlsx"),
        index=False,
    )

    # プロット
    plt.figure()
    plt.plot(x_axis, evals_result["valid_0"][metric], label="Validation")
    plt.legend()
    plt.xlabel("iteration")
    plt.ylabel(f"{metric}")
    plt.title(f"Learning curve for {metric} (Fold {fold + 1})")
    plt.savefig(
        os.path.join(output_dir, f"learning_curve_fold_{fold + 1}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_roc_curve(fpr, tpr, roc_auc, fold, output_dir):
    # DataFrameに変換
    df_roc_curve = pd.DataFrame(
        {
            "fpr": fpr,
            "tpr": tpr,
        }
    )

    # Excelファイルとして保存
    df_roc_curve.to_excel(
        os.path.join(output_dir, f"roc_curve_fold_{fold + 1}.xlsx"), index=False
    )

    # プロット
    plt.figure()
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic (Fold {fold + 1})")
    plt.legend(loc="lower right")
    plt.savefig(
        os.path.join(output_dir, f"roc_curve_fold_{fold + 1}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def train_and_evaluate_model(
    X, y, params, num_iterations, early_stopping_round, skf, output_dir
):
    (
        accuracies,
        precisions,
        recalls,
        f1s,
        f2s,
        log_losses,
        optimal_thresholds,
        best_iterations,
        roc_aucs,
        pr_aucs,
        mccs,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    last_evals_result = None
    last_roc_curve_data = None

    for fold, (train_index, valid_index) in enumerate(
        tqdm(skf.split(X, y), total=skf.get_n_splits(), desc="Cross-Validation")
    ):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

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

        # 確率値を取得
        y_prob = model.predict(X_valid, num_iteration=model.best_iteration)
        # ROC曲線をプロット
        fpr, tpr, thresholds = roc_curve(y_valid, y_prob)
        # PR曲線をプロット
        precision, recall, thresholds = precision_recall_curve(y_valid, y_prob)

        # コストベネフィット分析による最適閾値の決定

        # 偽陽性と偽陰性のコスト比率を定義
        c_fp = 100  # 偽陽性のコスト (賭け金額)
        c_fn = 1000  # 偽陰性のコスト (失われた利益)
        # 期待されるコストを計算
        expected_costs = []
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            fp = np.sum((y_pred == 1) & (y_valid == 0))
            fn = np.sum((y_pred == 0) & (y_valid == 1))
            cost = c_fp * fp + c_fn * fn
            expected_costs.append(cost)
        # 期待されるコストが最小となる閾値を選択
        optimal_threshold = thresholds[np.argmin(expected_costs)]
        optimal_thresholds.append(optimal_threshold)

        # クラス分類
        y_pred = [1 if x >= optimal_threshold else 0 for x in y_prob]

        accuracies.append(accuracy_score(y_valid, y_pred))
        precisions.append(precision_score(y_valid, y_pred, average="binary"))
        recalls.append(recall_score(y_valid, y_pred, average="binary"))
        f1s.append(f1_score(y_valid, y_pred, average="binary"))
        f2s.append(fbeta_score(y_valid, y_pred, beta=2, average="binary"))
        log_losses.append(log_loss(y_valid, y_prob))
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)
        pr_auc = auc(recall, precision)
        pr_aucs.append(pr_auc)
        mccs.append(matthews_corrcoef(y_valid, y_pred))

        # 最終foldのデータを記録
        if fold == skf.get_n_splits() - 1:
            last_evals_result = evals_result
            last_roc_curve_data = (fpr, tpr, roc_auc, fold)

    # 最終foldの学習曲線をプロット
    if last_evals_result:
        plot_learning_curve(last_evals_result, params["metric"], fold, output_dir)

    # 最終foldのROC曲線をプロット
    if last_roc_curve_data:
        fpr, tpr, roc_auc, fold = last_roc_curve_data
        plot_roc_curve(fpr, tpr, roc_auc, fold, output_dir)

    return (
        accuracies,
        precisions,
        recalls,
        f1s,
        f2s,
        log_losses,
        optimal_thresholds,
        best_iterations,
        roc_aucs,
        pr_aucs,
        mccs,
    )


def calculate_metrics(
    accuracies,
    precisions,
    recalls,
    f1s,
    f2s,
    mccs,
    log_losses,
    roc_aucs,
    pr_aucs,
    optimal_thresholds,
    best_iterations,
):
    metrics = {
        "accuracy": (np.mean(accuracies), np.std(accuracies)),
        "precision": (np.mean(precisions), np.std(precisions)),
        "recall": (np.mean(recalls), np.std(recalls)),
        "f1": (np.mean(f1s), np.std(f1s)),
        "f2": (np.mean(f2s), np.std(f2s)),
        "mcc": (np.mean(mccs), np.std(mccs)),
        "log_loss": (np.mean(log_losses), np.std(log_losses)),
        "roc_auc": (np.mean(roc_aucs), np.std(roc_aucs)),
        "pr_auc": (np.mean(pr_aucs), np.std(pr_aucs)),
        "optimal_threshold": (np.mean(optimal_thresholds), np.std(optimal_thresholds)),
        "best_iteration": (np.mean(best_iterations), np.std(best_iterations)),
    }
    return metrics


def display_metrics(metrics):
    for metric, (mean, std) in metrics.items():
        print(f"CV {metric.replace('_', ' ').title()}: {mean:.4f} ± {std:.4f}")


def calculate_feature_importance(model, feature_names, output_dir, top_n=20):
    importance = model.feature_importance(importance_type="gain")
    feature_importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importance}
    ).sort_values(by="Importance", ascending=False)

    # Excel ファイルとして保存
    feature_importance_df.to_excel(
        os.path.join(output_dir, "feature_importance.xlsx"), index=False
    )

    # プロット
    plt.figure(figsize=(10, 10))
    sns.barplot(x="Importance", y="Feature", data=feature_importance_df.head(top_n))
    plt.title(f"Top {top_n} Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.savefig(
        os.path.join(output_dir, f"feature_importance_top_{top_n}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    return feature_importance_df


def calculate_shap_values(model, X, output_dir):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # SHAP詳細プロット
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(
        os.path.join(output_dir, "shap_summary_plot.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Waterfallプロット (最も影響の大きい5つ)
    for i in range(min(5, len(X))):
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[i],
                base_values=explainer.expected_value,
                data=X.iloc[i],
            ),
            show=False,
        )
        plt.savefig(
            os.path.join(output_dir, f"shap_waterfall_plot_{i}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    return shap_values


def create_final_model(X, y, params, best_iterations):
    final_train_data = lgb.Dataset(X, label=y)
    final_num_iteration = (
        int(np.mean(best_iterations)) if int(np.mean(best_iterations)) > 50 else 50
    )
    final_model = lgb.train(
        params=params, train_set=final_train_data, num_boost_round=final_num_iteration
    )
    return final_model


def prepare_test_data(df_test_features, key_columns, final_model, mean_best_threshold):
    X_test = df_test_features.drop(key_columns, axis=1)
    y_test_pred = final_model.predict(X_test, num_iteration=final_model.best_iteration)
    y_test_pred_classes = [1 if x >= mean_best_threshold else 0 for x in y_test_pred]
    return y_test_pred_classes, y_test_pred


def merge_results(
    df_test_features, y_test_pred_classes, y_test_pred, df_all, target_col
):
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

    df["発走日時"] = df["IN_年月日"] + df["IN_発走時間"]
    df["発走日時"] = pd.to_datetime(df["発走日時"], format="%Y%m%d%H%M")

    df["IN_馬番"] = df["IN_馬番"].astype(int)
    df["IN_枠番"] = df["IN_枠番"].astype(int)
    df[f"{target_col}予想"] = df[f"{target_col}予想"].astype(int)
    df[f"{target_col}確率"] = df[f"{target_col}確率"].astype(float)

    df = df.sort_values(by=["IN_レースキー_場コード", "IN_レースキー_R", "IN_馬番"])

    df = df[
        [
            "レースキー",
            "IN_レースキー_場コード",
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

    column_names = [
        "開催情報",
        "レース場",
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

    return df


# シードの設定
RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)

# 環境変数の読み込み
project_path = "../../"
env_file = os.getenv("ENV_FILE", os.path.join(project_path, ".env"))
load_dotenv(env_file)
file_directory = os.getenv("DF_DIR")
pred_directory = os.path.join(str(os.getenv("PRED_DIR")), "BINARY")
matplotlib_fontja.japanize()

# 結果出力ディレクトリ作成
os.makedirs(pred_directory, exist_ok=True)

# データの読み込み
df_all, df_target, df_features, df_test_features = read_data(file_directory)

# 目的変数を選択
target_col = "穴馬"
key_columns = [
    "IN_レースキー",
    "IN_レースキー_場コード",
    "IN_レースキー_回",
    "IN_レースキー_日",
    "IN_レースキー_R",
    "_merge",
]

# 説明変数と目的変数の分割
X, y = prepare_data(df_features, df_target, key_columns, target_col)

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
    "is_unbalance": "true",
    "min_data_in_leaf": 30,  # default: 20, increase if necessary
    "min_sum_hessian_in_leaf": 1e-2,  # default: 1e-3, increase if necessary
}
num_iterations = 10000
early_stopping_round = 100

# Stratified K-Fold Cross-Validationの設定
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

# 交差検証の実行
(
    accuracies,
    precisions,
    recalls,
    f1s,
    f2s,
    log_losses,
    optimal_thresholds,
    best_iterations,
    roc_aucs,
    pr_aucs,
    mccs,
) = train_and_evaluate_model(
    X, y, params, num_iterations, early_stopping_round, skf, pred_directory
)

# 評価結果の表示
metrics = calculate_metrics(
    accuracies,
    precisions,
    recalls,
    f1s,
    f2s,
    mccs,
    log_losses,
    roc_aucs,
    pr_aucs,
    optimal_thresholds,
    best_iterations,
)
display_metrics(metrics)

# 全データを用いて最終モデルを再学習する
final_model = create_final_model(X, y, params, best_iterations)

# Feature Importance の計算とプロット
feature_importance_df = calculate_feature_importance(
    final_model, X.columns, pred_directory
)

# SHAP値の計算とプロット
shap_values = calculate_shap_values(final_model, X, pred_directory)

# テストデータの予測
y_test_pred_classes, y_test_pred = prepare_test_data(
    df_test_features, key_columns, final_model, metrics["optimal_threshold"][0]
)

# 結果のマージ
df_result = merge_results(
    df_test_features, y_test_pred_classes, y_test_pred, df_all, target_col
)

# カラムの変換
df_transformed = transform_columns(df_result)

# 結果の保存
df_transformed.to_excel(os.path.join(pred_directory, "result.xlsx"), index=False)
