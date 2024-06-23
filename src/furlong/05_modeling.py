import os

import matplotlib.pyplot as plt
import numpy as np
import optuna.integration.lightgbm as lgb
import pandas as pd
import seaborn as sns
import shap
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold


def plot_learning_curve(model, metric="multi_logloss"):
    results = model.eval_valid()
    metric_values = results[metric]

    plt.figure(figsize=(10, 6))
    plt.plot(metric_values, label=f"Validation {metric}")
    plt.xlabel("Iteration")
    plt.ylabel(metric)
    plt.title(f"Learning Curve ({metric})")
    plt.legend()
    plt.show()


def plot_roc_curve(y_true, y_pred, num_classes):
    plt.figure(figsize=(10, 6))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"Class {i} ROC curve (area = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()


# 環境変数の読み込み
project_path = "../../"
env_file = os.getenv("ENV_FILE", os.path.join(project_path, ".env"))
load_dotenv(env_file)
file_directory = os.getenv("DF_DIR")

# データの読込み
df_target = pd.read_feather(os.path.join(str(file_directory), "DF_TARGET_trim.feather"))
df_features = pd.read_feather(
    os.path.join(str(file_directory), "DF_FEATURES_trim.feather")
)

# 目的変数を選択
target_col = "順位カテゴリ"
# 学習に使わないカラム
key_columns = [
    "IN_レースキー_場コード",
    "IN_レースキー_年",
    "IN_レースキー_回",
    "IN_レースキー_日",
    "IN_レースキー_R",
]
# 説明変数と目的変数の分割
X = df_features.drop(key_columns, axis=1)
y = df_target[target_col]

# ハイパーパラメータチューニング
params = {
    "objective": "multiclass",  # 多値分類の場合
    "metric": "multi_logloss",  # 多値分類の場合
    "boosting_type": "gbdt",
    "num_class": len(y.unique()),  # クラス数を指定
    "device": "cpu",
    "max_bin": 255,  # ビンのサイズを減らす
}
tuner = lgb.LightGBMTunerCV(
    params,
    lgb.Dataset(X, label=y),
    num_boost_round=1000,
    nfold=5,
    return_cvbooster=True,
)
tuner.run()

# 最良のハイパーパラメータ
best_params = tuner.best_params
best_booster = tuner.get_best_booster()
best_iteration = (
    best_booster.best_iteration if best_booster.best_iteration > 0 else 100
)  # デフォルト値が100
print(f"Best num_boost_round: {best_iteration}")
print(f"Best parameters: {best_params}")
print(f"Best score: {tuner.best_score}")

# StratifiedKFoldによる交差検証
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
log_losses = []
precisions = []
recalls = []
f1s = []

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # モデルの再訓練
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    gbm = lgb.train(
        best_params,
        train_data,
        num_boost_round=best_booster.best_iteration,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
    )

    # モデルの評価
    preds = gbm.predict(X_test)
    pred_labels = [np.argmax(p) for p in preds]

    accuracies.append(accuracy_score(y_test, pred_labels))
    log_losses.append(log_loss(y_test, preds))
    precisions.append(precision_score(y_test, pred_labels, average="weighted"))
    recalls.append(recall_score(y_test, pred_labels, average="weighted"))
    f1s.append(f1_score(y_test, pred_labels, average="weighted"))

# 交差検証の結果を表示
print(f"Cross-validated Accuracy: {np.mean(accuracies)} ± {np.std(accuracies)}")
print(f"Cross-validated Log Loss: {np.mean(log_losses)} ± {np.std(log_losses)}")
print(f"Cross-validated Precision: {np.mean(precisions)} ± {np.std(precisions)}")
print(f"Cross-validated Recall: {np.mean(recalls)} ± {np.std(recalls)}")
print(f"Cross-validated F1 Score: {np.mean(f1s)} ± {np.std(f1s)}")

# 学習曲線の表示
plot_learning_curve(gbm, metric="multi_logloss")

# ROC曲線の表示
plot_roc_curve(y_test, np.array(preds), len(y.unique()))

# 特徴量重要度の取得
importance = gbm.feature_importance()
features = X.columns

# データフレームに変換
feature_importance_df = pd.DataFrame(
    {"Feature": features, "Importance": importance}
).sort_values(by="Importance", ascending=False)

# 特徴量重要度のプロット
plt.figure(figsize=(10, 8))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
plt.title("Feature Importance")
plt.show()

# SHAP explainerの作成
explainer = shap.TreeExplainer(gbm)
shap_values = explainer.shap_values(X_test)

# SHAP summary plotの表示
shap.summary_plot(shap_values, X_test)

# SHAP dependence plotの表示
shap.dependence_plot("特定の特徴量の名前", shap_values, X_test)
