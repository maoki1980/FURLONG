import json
import os
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo

import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib_fontja
import numpy as np
import optuna
import pandas as pd
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
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


# ディレクトリを作成する関数
def create_directory(path):
    try:
        os.makedirs(path, exist_ok=True)
        # ディレクトリの存在を確認
        if not os.path.isdir(path):
            raise RuntimeError(f"Directory creation failed: {path}")
    except Exception as e:
        print(f"Error: {e}")
        raise


# JSTで現在日時を取得して所定のフォーマットで返す関数
def get_tokyo_time_formatted():
    tokyo_time = datetime.now(ZoneInfo("Asia/Tokyo"))
    return tokyo_time.strftime("%Y%m%d_%H%M%S")


# G-Meanを算出する関数
def geometric_mean(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return np.sqrt(sensitivity * specificity)


# オーバーサンプリングする関数
def perform_oversampling(X, y, majority_fractor=0.3):
    logger.debug(f"Original training dataset's label count: {Counter(y)}")
    logger.debug("Resampling start.")
    class_counts = y.value_counts()
    majority_class_count = class_counts[0]
    sampling_strategy = {1: int(majority_class_count * majority_fractor)}

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=RANDOM_SEED)
    X_res, y_res = smote.fit_resample(X, y)
    logger.debug("Resampling finished.")
    logger.debug(f"Resampled training dataset's label count: {Counter(y_res)}")

    return X_res, y_res


# コストベネフィット分析による最適閾値を決定する関数
def find_optimal_threshold(y_valid, y_proba, thresholds):
    expected_costs = []  # 期待されるコスト
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        fp = np.sum((y_pred == 1) & (y_valid == 0))
        fn = np.sum((y_pred == 0) & (y_valid == 1))
        cost = C_FP * fp + C_FN * fn
        expected_costs.append(cost)

    # 期待されるコストが最小となる閾値を選択
    optimal_threshold = thresholds[np.argmin(expected_costs)]

    return optimal_threshold


# Optunaを用いた交差検証によるハイパーパラメーター最適化関数
def objective(trial):
    # ハイパーパラメーター
    target_params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.5, log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 128),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0, step=0.1),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0, step=0.1),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 500),
    }
    # パラメーターの設定
    params = BASE_PARAMS.copy()
    params.update(target_params)

    # 層化k分割交差検証を行う
    skf = StratifiedKFold(n_splits=FOLD_NO, shuffle=True, random_state=RANDOM_SEED)
    # 評価指標の初期化
    mcc_scores = []
    num_iterations = []

    # k分割分繰り返す
    for i, (train_index, valid_index) in enumerate(skf.split(X, y)):
        # データを分割
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        # オーバーサンプリングを適用
        if RESAMPLING:
            X_train_res, y_train_res = perform_oversampling(X_train, y_train)
        else:
            X_train_res, y_train_res = X_train, y_train
        # LightGBM用のデータセットを作成
        train_data = lgb.Dataset(X_train_res, label=y_train_res)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
        # 不均衡比率の計算
        ratio = Counter(y_train_res)[0] / Counter(y_train_res)[1]
        # paramsのアップデート
        params.update({"scale_pos_weight": ratio})

        # モデル学習
        model = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[valid_data],
        )

        # 予測を行い確率値を取得
        y_proba = model.predict(X_valid, num_iteration=model.best_iteration)
        # PR曲線を取得
        precision, recall, thresholds = precision_recall_curve(y_valid, y_proba)
        # コストベネフィット分析により期待されるコストが最小となる閾値を選択
        opt_thre = find_optimal_threshold(y_valid, y_proba, thresholds)
        # 確率値からクラスに分類
        y_pred = [1 if prob >= opt_thre else 0 for prob in y_proba]
        # マシューズ相関係数 (MCC) の計算
        mcc = matthews_corrcoef(y_valid, y_pred)
        mcc_scores.append(mcc)
        # イテレーション数
        best_iter = model.best_iteration
        num_iterations.append(best_iter)

        # デバッグログの出力
        logger.debug(
            f"Fold {i+1}, MCC score: {mcc:.4f} on thres: {opt_thre:.4f}, iter: {best_iter}"
        )

    # 全Foldの平均を計算
    mean_mcc = np.mean(mcc_scores)
    mean_iteration = np.mean(num_iterations)

    # ユーザー属性として保存
    trial.set_user_attr("num_iteration", mean_iteration)

    return mean_mcc


# 最適化したパラメーター一式をJSONファイルで保存する関数
def save_trial_params(study, trial):
    # 固定値のパラメータを含めた全体のパラメータセットを作成
    full_params = BASE_PARAMS.copy()
    full_params.update(trial.params)
    result = {
        "results": {
            "trial_number": trial.number,
            "mean_score": trial.value,
            "mean_iteration": trial.user_attrs.get("num_iteration"),
        }
    }
    full_params.update(result)
    # 試行回数の文字列を作成
    trial_number = str(trial.number).zfill(3)
    # JSON形式で保存
    with open(
        os.path.join(param_dir, f"params_{START_DATETIME}_trial_{trial_number}.json"),
        "w",
    ) as f:
        json.dump(full_params, f, indent=4)


def print_metrics(summary, suffix):
    print(f"{suffix}'s train: ====================================================")
    for key in summary.index:
        metric_name = key.replace("_", " ").title()
        mean_value = summary.at[key, "mean"]
        std_dev = summary.at[key, "std_dev"]

        formatted_mean = f"{mean_value:.4f}"
        formatted_std_dev = f"{std_dev:.4f}"

        print(
            f"    {FOLD_NO}-fold CV {metric_name}: {formatted_mean} ± {formatted_std_dev}"
        )
    print("==============================================================: Output end.")


# 学習曲線プロットを出力する関数
def plot_learning_curve(learning_curve_data, metric, fold, output_dir):
    iterations = len(learning_curve_data["valid_0"][metric])
    x_axis = range(0, iterations)
    # DataFrameに変換
    df_learning_curve = pd.DataFrame(
        {
            "iteration": x_axis,
            "validation": learning_curve_data["valid_0"][metric],
        }
    )
    # Excelファイルとして保存
    df_learning_curve.to_excel(
        os.path.join(
            output_dir, f"learning_curve_{START_DATETIME}_fold_{fold + 1}.xlsx"
        ),
        index=False,
    )
    # プロット
    plt.figure()
    plt.plot(x_axis, learning_curve_data["valid_0"][metric], label="Validation")
    plt.xlabel("Iteration")
    plt.ylabel(f"{metric}")
    plt.title(f"Learning Curve for {metric} (Fold {fold + 1})")
    plt.legend()
    plt.savefig(
        os.path.join(
            output_dir, f"learning_curve_{START_DATETIME}_fold_{fold + 1}.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# ROC曲線プロットを出力する関数
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
        os.path.join(output_dir, f"roc_curve_{START_DATETIME}_fold_{fold + 1}.xlsx"),
        index=False,
    )
    # プロット
    plt.figure()
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver-Operating-Characteristic Curve (Fold {fold + 1})")
    plt.legend()
    plt.savefig(
        os.path.join(output_dir, f"roc_curve_{START_DATETIME}_fold_{fold + 1}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# PR曲線プロットを出力する関数
def plot_pr_curve(precision, recall, pr_auc, fold, output_dir):
    # DataFrameに変換
    df_pr_curve = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
        }
    )
    # Excelファイルとして保存
    df_pr_curve.to_excel(
        os.path.join(output_dir, f"pr_curve_{START_DATETIME}_fold_{fold + 1}.xlsx"),
        index=False,
    )
    # プロット
    plt.figure()
    plt.plot(
        recall, precision, color="blue", lw=2, label=f"PR curve (area = {pr_auc:.2f})"
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (Fold {fold + 1})")
    plt.legend()
    plt.savefig(
        os.path.join(output_dir, f"pr_curve_{START_DATETIME}_fold_{fold + 1}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# 定数を定義
RANDOM_SEED = 0  # シードの設定
START_DATETIME = get_tokyo_time_formatted()  # 現在日時
FOLD_NO = 5  # 交差検証のFold数
C_FP = 100  # 偽陽性のコスト (賭け金額)
C_FN = 500  # 偽陰性のコスト (失われた利益)
BASE_PARAMS = {
    "task": "train",
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting": "gbdt",
    "seed": RANDOM_SEED,
    "bagging_seed": RANDOM_SEED,
    "feature_fraction_seed": RANDOM_SEED,
    "extra_seed": RANDOM_SEED,
    "verbosity": 1,
    "early_stopping_round": 20,
    "device_type": "gpu",
    "num_leaves": 31,  # default: 31
    "min_data_in_leaf": 20,  # default: 20
    "lambda_l1": 0.1,  # default: 0.0
    "lambda_l2": 0.1,  # default: 0.0
}
NUM_BOOST_ROUND = 10000  # 最大イテレーション数
RESAMPLING = 1  # リサンプリングするか
CV = 1  # 交差検証するか
OPTIMIZE = 1  # 最適化するか

# 環境変数の読み込み
project_path = "../../"
env_file = os.getenv("ENV_FILE", os.path.join(project_path, ".env"))
load_dotenv(env_file)
file_dir = os.getenv("DF_DIR")
pred_dir = os.path.join(str(os.getenv("PRED_DIR")), "BINARY")
param_dir = os.path.join(str(pred_dir), "PARAMS")
# 初期設定
matplotlib_fontja.japanize()
np.random.seed(RANDOM_SEED)
logger.remove()  # デフォルトのログ出力を削除
logger.add(
    lambda msg: print(msg, end=""),
    format="[{time:YYYY-MM-DD HH:mm:ss}][{level}] {message}",
)
os.environ["LOKY_MAX_CPU_COUNT"] = "10"  # 使用するコア数を指定
read_json_name = "optimized_params_fixed.json"

# 結果出力ディレクトリ作成
create_directory(pred_dir)
create_directory(param_dir)

# データの読み込み
df_all = pd.read_feather(os.path.join(str(file_dir), "ALL_DATA.feather"))
df_target = pd.read_feather(os.path.join(str(file_dir), "DF_TARGET.feather"))
df_features = pd.read_feather(os.path.join(str(file_dir), "DF_FEATURES.feather"))
df_test = pd.read_feather(os.path.join(str(file_dir), "DF_TEST_FEATURES.feather"))

# 目的変数を選択
target_col = "複勝穴馬"
key_columns = [
    "IN_レースキー",
    "IN_レースキー_場コード",
    "IN_レースキー_回",
    "IN_レースキー_日",
    "IN_レースキー_R",
    "_merge",
]

# 説明変数と目的変数の分割
df_data = pd.concat(
    [df_features.drop(key_columns, axis=1), df_target[target_col]], axis=1
).copy()
X = df_data.drop(target_col, axis=1)
y = df_data[target_col]
logger.info(f"Original dataset's label count: {Counter(y)}")

# ハイパーパラメーターをファイルから読み込む
json_path = os.path.join(param_dir, read_json_name)
if os.path.exists(json_path):
    # 最適化済みパラメーターを読み込む
    with open(json_path, "r") as f:
        loaded_params = json.load(f)
    loaded_params.pop("results", None)
    params = loaded_params.copy()
    logger.info(f"Loaded parameters from {read_json_name} file.")
    read_json = 1
else:
    # デフォルトパラメーターを設定する
    params = BASE_PARAMS.copy()
    logger.info("Set default parameters.")
    read_json = 0

# 交差検証を行う
if CV:
    logger.info("Cross-Validation: ON")
    # 層化k分割交差検証を行う
    skf = StratifiedKFold(n_splits=FOLD_NO, shuffle=True, random_state=RANDOM_SEED)

    # 評価指標の結果を初期化
    results = []

    # k分割分だけ繰り返す
    logger.info("Cross-Varidation start.")
    last_learning_curve, last_roc_curve, last_pr_curve = None, None, None
    for fold, (train_index, valid_index) in enumerate(skf.split(X, y)):
        # データを分割
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        # オーバーサンプリングを適用
        if RESAMPLING:
            X_train_res, y_train_res = perform_oversampling(X_train, y_train)
        else:
            X_train_res, y_train_res = X_train, y_train
        # LightGBM用のデータセットを作成
        train_data = lgb.Dataset(X_train_res, label=y_train_res)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
        # 不均衡比率の計算
        ratio = Counter(y_train_res)[0] / Counter(y_train_res)[1]
        # paramsのアップデート
        params.update({"learning_rate": 0.1 / ratio})
        params.update({"scale_pos_weight": ratio})

        # モデル学習
        eval_result = {}
        model = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[valid_data],
            callbacks=[
                lgb.record_evaluation(eval_result=eval_result),
            ],
        )

        # 予測を行い確率値を取得
        y_proba = model.predict(X_valid, num_iteration=model.best_iteration)

        # ROC曲線を取得
        fpr, tpr, thresholds = roc_curve(y_valid, y_proba)
        # PR曲線を取得
        precision, recall, thresholds = precision_recall_curve(y_valid, y_proba)

        # コストベネフィット分析により期待されるコストが最小となる閾値を選択
        opt_thre = find_optimal_threshold(y_valid, y_proba, thresholds)

        # 確率値からクラスに分類
        y_pred = [1 if prob >= opt_thre else 0 for prob in y_proba]

        # 評価指標を取得
        mcc = matthews_corrcoef(y_valid, y_pred)
        best_iter = model.best_iteration

        # 評価指標を計算して結果を辞書に保存
        results.append(
            {
                # Log損失 (Log loss)
                "log_loss": log_loss(y_valid, y_proba),
                # 正解率 (Accuracy)
                "accuracy": accuracy_score(y_valid, y_pred),
                # 適合率 (Precision)
                "precision": precision_score(y_valid, y_pred, average="binary"),
                # 再現率 (Recall)
                "recall": recall_score(y_valid, y_pred, average="binary"),
                # F1-score
                "f1_score": f1_score(y_valid, y_pred, average="binary"),
                # Fbeta-score
                "f2_score": fbeta_score(y_valid, y_pred, beta=2, average="binary"),
                # G-Mean
                "g_mean": geometric_mean(y_valid, y_pred),
                # マシューズ相関係数 (Matthews Correlation Coefficient; MCC)
                "mcc_score": mcc,
                # ROC-AUC (Receiver Operating Characteristic - Area Under the Curve; ROC曲線)
                "roc_auc": auc(fpr, tpr),
                # PR-AUC (Precision-Recall - Area Under the Curve; PR曲線)
                "pr_auc": auc(recall, precision),
                # 最適化した閾値
                "optimal_threshold": opt_thre,
                # イテレーション数
                "num_iterations": best_iter,
            }
        )
        # 最終foldのデータを記録
        if fold == skf.get_n_splits() - 1:
            last_learning_curve = (eval_result, fold)
            last_roc_curve = (fpr, tpr, auc(fpr, tpr), fold)
            last_pr_curve = (precision, recall, auc(recall, precision), fold)
        # ログ出力
        logger.info(
            f"Fold {fold+1}, MCC score: {mcc:.4f} on thre: {opt_thre:.4f}, iter {best_iter}"
        )
    # 最終foldの学習曲線をプロット
    if last_learning_curve:
        learning_curve_data, fold = last_learning_curve
        plot_learning_curve(learning_curve_data, params["metric"], fold, pred_dir)
    # 最終foldのROC曲線をプロット
    if last_roc_curve:
        fpr, tpr, roc_auc, fold = last_roc_curve
        plot_roc_curve(fpr, tpr, roc_auc, fold, pred_dir)
    # 最終foldのPR曲線をプロット
    if last_pr_curve:
        precision, recall, pr_auc, fold = last_pr_curve
        plot_pr_curve(precision, recall, pr_auc, fold, pred_dir)

    # 各評価指標の平均と標準偏差を計算
    results_df = pd.DataFrame(results)
    summary = results_df.describe().transpose()[["mean", "std"]]
    summary.columns = ["mean", "std_dev"]

    # Excelファイルに保存
    suffix = "optimalParams" if read_json == 1 else "defaultParams"
    with pd.ExcelWriter(
        os.path.join(pred_dir, f"CV_{START_DATETIME}_metrics_{suffix}.xlsx")
    ) as writer:
        summary.to_excel(writer, sheet_name="Summary")
        results_df.to_excel(writer, sheet_name="Detail", index=False)

    # 各評価指標をprintする
    print_metrics(summary, suffix)
    # ログ出力
    logger.info("Cross-Varidation finished.")
else:
    logger.info("Cross-Validation: OFF")

# Oputunaでハイパーパラメーターを最適化する (MCCを最大化)
if OPTIMIZE:
    logger.info("Hyper parameter optimize: ON")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1, callbacks=[save_trial_params])
    # 最適化結果の表示
    print("Optimal train: ============================================================")
    print("Best trial:")
    trial = study.best_trial
    print(f"    Trial Number: {trial.number}")
    print(
        f"    {FOLD_NO}-fold CV Num of iteration: {trial.user_attrs.get('num_iteration')}"
    )
    print(f"    {FOLD_NO}-fold CV MCC Score: {trial.value:.4f}")
    print("    Params:")
    for key, value in trial.params.items():
        print(f"        {key}: {value}")
    print("==============================================================: Output end.")
    result = {
        "results": {
            "trial_number": trial.number,
            "mean_score": trial.value,
            "mean_iteration": trial.user_attrs.get("num_iteration"),
        }
    }
    # 最適化したハイパーパラメーターをJSONで保存
    optimized_params = BASE_PARAMS.copy()
    optimized_params.update(trial.params)
    optimized_params.update(result)
    with open(
        os.path.join(param_dir, f"optimized_params_{START_DATETIME}.json"), "w"
    ) as f:
        json.dump(optimized_params, f, indent=4)
else:
    logger.info("Hyper parameter optimize: OFF")

# 全データを用いて最終モデルを学習する
logger.info("Create final model.")
if RESAMPLING:
    X_res, y_res = perform_oversampling(X, y)
else:
    X_res, y_res = X, y
final_train_data = lgb.Dataset(X_res, label=y_res)
if CV:
    final_num_iteration = int(summary.loc["num_iterations", "mean"])
else:
    final_num_iteration = NUM_BOOST_ROUND
logger.info(f"Set number of iteration: {final_num_iteration}")
params.pop("early_stopping_round", None)
final_model = lgb.train(
    params=params,
    train_set=final_train_data,
    num_boost_round=final_num_iteration,
)

# 推定
X_test = df_test.drop(key_columns, axis=1)
y_test_proba = final_model.predict(X_test, num_iteration=final_model.best_iteration)
if CV:
    best_threshold = summary.loc["optimal_threshold", "mean"]
else:
    best_threshold = 0.5
y_test_pred = (y_test_proba > best_threshold).astype(int)
