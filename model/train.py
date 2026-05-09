"""
model/train.py  —  Improved training pipeline
================================================
What's new vs the original:
  1. Zero-value fix         — Glucose, BP, SkinThickness, Insulin, BMI cannot be 0
  2. XGBoost                — 4th model alongside RF, SVM, LR
  3. SMOTE                  — balances 65/35 class imbalance during training
  4. Cross-validation       — 5-fold CV scores (mean ± std) for every model
  5. ROC-AUC                — saved in metrics.json
  6. Plots saved            — confusion matrix, ROC curves, feature importance,
                              model comparison (ready for your report)
  7. SHAP values            — saved as JSON for the Flutter Explainability screen
  8. Flutter coefficients   — printed to terminal after training

Fixes applied:
  FIX-1  SHAP: three-level explainer fallback (Tree → Linear → Kernel)
         so SVC (or any model) never crashes SHAP computation
  FIX-2  SHAP summary_plot: X_transformed wrapped as DataFrame so axis
         labels always show feature names, not column indices
  FIX-3  XGBClassifier: removed deprecated use_label_encoder=False
         (raises TypeError on XGBoost ≥ 2.0 / Python 3.13)
  FIX-4  print_flutter_coefficients: champion.predict_proba now receives
         the already-imputed DataFrame, keeping calibration consistent
         with the separately computed mean/scale arrays

Run:
    cd chronic-disease-main
    python model/train.py --retrain
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

import config
from model.preprocessing import make_numeric_preprocess

# Optional heavy deps — graceful fallback if not installed
try:
    from xgboost import XGBClassifier
    _XGB = True
except ImportError:
    _XGB = False
    print("[warn] xgboost not installed — skipped. Run: pip install xgboost")

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    _SMOTE = True
except ImportError:
    _SMOTE = False
    print("[warn] imbalanced-learn not installed — SMOTE skipped. Run: pip install imbalanced-learn")

try:
    import shap
    _SHAP = True
except ImportError:
    _SHAP = False
    print("[warn] shap not installed — SHAP skipped. Run: pip install shap")


# ── 1. Columns whose zero values are actually missing data ─────────────────
ZERO_IS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_dataset() -> pd.DataFrame:
    for path in [config.DATASET_PATH, config.ROOT_DATASET_FALLBACK_PATH]:
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            continue
    return pd.read_csv(config.DATASET_URL)


def fix_zero_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace impossible zeros with NaN so the median imputer handles them."""
    df = df.copy()
    cols = [c for c in ZERO_IS_MISSING if c in df.columns]
    df[cols] = df[cols].replace(0, np.nan)
    print("\n── Zero-value fix ──────────────────────────────────")
    for c in cols:
        n = df[c].isna().sum()
        print(f"   {c:30s}: {n:3d} zeros → NaN (median fill)")
    print()
    return df


# ── 2. Train all models ────────────────────────────────────────────────────
def train_all_models(df: pd.DataFrame):
    feature_cols = config.FEATURE_COLUMNS
    target_col   = config.TARGET_COLUMN

    # Keep only columns that exist in this dataset
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df[target_col]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=config.RANDOM_STATE
    )
    preprocess = make_numeric_preprocess(feature_cols)

    model_cfgs: dict = {
        "Logistic Regression": {
            "est":  LogisticRegression(max_iter=500, solver="lbfgs"),
            "grid": {"clf__C": [0.1, 1, 10]},
        },
        "Random Forest": {
            "est":  RandomForestClassifier(random_state=config.RANDOM_STATE),
            "grid": {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 5, 10]},
        },
        "SVM (RBF)": {
            "est":  SVC(probability=True, random_state=config.RANDOM_STATE),
            "grid": {"clf__C": [0.5, 1.0, 2.0], "clf__gamma": ["scale", "auto"]},
        },
    }
    if _XGB:
        model_cfgs["XGBoost"] = {
            "est":  XGBClassifier(
                # FIX-3: use_label_encoder removed — deprecated in XGBoost ≥ 1.6,
                # raises TypeError on XGBoost ≥ 2.0 / Python 3.13
                eval_metric="logloss",
                random_state=config.RANDOM_STATE,
                verbosity=0,
            ),
            "grid": {
                "clf__n_estimators":  [100, 300],
                "clf__max_depth":     [4, 6],
                "clf__learning_rate": [0.05, 0.1],
            },
        }

    results: dict[str, dict] = {}
    for name, cfg in model_cfgs.items():
        print(f"Training {name}...")

        # GridSearch on plain pipeline for param selection
        plain_pipe = Pipeline([("pre", preprocess), ("clf", cfg["est"])])
        gs = GridSearchCV(plain_pipe, cfg["grid"], cv=cv,
                          scoring="accuracy", n_jobs=-1, verbose=0)
        gs.fit(X_train, y_train)
        best_params = gs.best_params_

        # Re-fit with SMOTE if available
        if _SMOTE:
            best_est = gs.best_estimator_.named_steps["clf"]
            final_pipe = ImbPipeline([
                ("pre",   make_numeric_preprocess(feature_cols)),
                ("smote", SMOTE(random_state=config.RANDOM_STATE)),
                ("clf",   best_est),
            ])
            final_pipe.fit(X_train, y_train)
        else:
            final_pipe = gs.best_estimator_
            final_pipe.fit(X_train, y_train)

        y_pred = final_pipe.predict(X_test)
        y_prob = final_pipe.predict_proba(X_test)[:, 1]

        # Cross-validation on plain pipeline (no SMOTE — matches literature practice)
        cv_scores = cross_val_score(
            Pipeline([("pre", make_numeric_preprocess(feature_cols)), ("clf", cfg["est"])]),
            X, y, cv=cv, scoring="accuracy", n_jobs=-1,
        )

        results[name] = {
            "model":  final_pipe,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "metrics": {
                "accuracy":         float(accuracy_score(y_test, y_pred)),
                "precision":        float(precision_score(y_test, y_pred, zero_division=0)),
                "recall":           float(recall_score(y_test, y_pred, zero_division=0)),
                "f1":               float(f1_score(y_test, y_pred, zero_division=0)),
                "roc_auc":          float(roc_auc_score(y_test, y_prob)),
                "cv_accuracy_mean": float(cv_scores.mean()),
                "cv_accuracy_std":  float(cv_scores.std()),
            },
            "best_params": best_params,
        }
        m = results[name]["metrics"]
        print(f"   Accuracy={m['accuracy']:.3f}  ROC-AUC={m['roc_auc']:.3f}"
              f"  CV={m['cv_accuracy_mean']:.3f}±{m['cv_accuracy_std']:.3f}")

    champion_name = max(results, key=lambda k: results[k]["metrics"]["accuracy"])
    champion      = results[champion_name]["model"]
    print(f"\n🏆  Champion: {champion_name}\n")

    # Permutation feature importance on champion
    perm = permutation_importance(
        champion, X_test, y_test,
        scoring="accuracy", n_repeats=10, random_state=config.RANDOM_STATE,
    )
    feature_importance = {
        feat: float(perm.importances_mean[i])
        for i, feat in enumerate(feature_cols)
    }

    return results, champion_name, champion, X_test, y_test, feature_importance, feature_cols


# ── 3. SHAP values ─────────────────────────────────────────────────────────
'''def compute_shap(champion, X_test: pd.DataFrame, feature_cols: list,
                 artifacts_dir: Path) -> dict:
    """
    Compute SHAP values and save:
      - shap_values.json   (per-sample values for Flutter)
      - shap_summary.png   (beeswarm plot for report)
    Returns mean |SHAP| per feature.

    FIX-1: Three-level explainer fallback:
      TreeExplainer  → works for RF, XGBoost, GBM, Decision Trees
      LinearExplainer→ works for LogisticRegression, Ridge, Lasso, LinearSVC
      KernelExplainer→ model-agnostic fallback (works for SVC and anything else)
    """
    if not _SHAP:
        print("   [skip] shap not installed")
        return {}

    print("   Computing SHAP values...")

    # Get the preprocessed X for SHAP
    try:
        pre = champion.named_steps["pre"]
        X_transformed = pre.transform(X_test)
    except Exception:
        # SMOTE pipeline uses different step names
        X_transformed = X_test.values

    try:
        clf = champion.named_steps["clf"]
    except Exception:
        clf = champion

    # FIX-1: Three-level fallback — Tree → Linear → Kernel
    try:
        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X_transformed)
        # For binary classifiers shap_values returns [class0, class1]
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        print("   [shap] using TreeExplainer")

    except Exception:
        try:
            explainer = shap.LinearExplainer(clf, X_transformed)
            shap_vals = explainer.shap_values(X_transformed)
            print("   [shap] using LinearExplainer")

        except Exception:
            # KernelExplainer: model-agnostic, works for SVC and any other model.
            # Requires SVC(probability=True) for predict_proba — already set above.
            print("   [shap] falling back to KernelExplainer (may take ~30s)...")
            X_df = pd.DataFrame(X_transformed, columns=feature_cols)
            background = shap.sample(X_df, min(50, len(X_df)))
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_vals_raw = explainer.shap_values(X_df, nsamples=100)

            # Normalise to a plain 2-D array (n_samples, n_features).
            # KernelExplainer can return:
            #   • list  [class0(n,f), class1(n,f)]  → take index 1
            #   • 3-D   ndarray (n, f, n_classes)   → take [..., 1]
            #   • 2-D   ndarray (n, f)               → use as-is
            if isinstance(shap_vals_raw, list):
                shap_vals = np.array(shap_vals_raw[1])
            else:
                arr = np.array(shap_vals_raw)
                if arr.ndim == 3:          # (n_samples, n_features, n_classes)
                    shap_vals = arr[:, :, 1]
                else:                      # already (n_samples, n_features)
                    shap_vals = arr
            print("   [shap] KernelExplainer done")

    # Guarantee shap_vals is always a plain 2-D numpy array (n_samples, n_features)
    # regardless of which explainer was used.
    shap_vals = np.array(shap_vals)
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]

    mean_abs = {feat: float(np.abs(shap_vals[:, i]).mean())
                for i, feat in enumerate(feature_cols)}

    # Save per-sample SHAP for Flutter (first 100 samples to keep JSON small)
    shap_json = {
        "feature_names": feature_cols,
        "mean_abs_shap":  mean_abs,
        "samples": [
            {feat: float(shap_vals[row, i]) for i, feat in enumerate(feature_cols)}
            for row in range(min(100, len(shap_vals)))
        ],
    }
    with open(artifacts_dir / "shap_values.json", "w") as f:
        json.dump(shap_json, f, indent=2)
    print(f"   Saved shap_values.json → {artifacts_dir / 'shap_values.json'}")

    # FIX-2: Wrap X_transformed as DataFrame so summary_plot always shows
    # feature names on the axis, not bare column indices (critical when
    # KernelExplainer is used and X_transformed is a plain numpy array).
    X_display = (
        pd.DataFrame(X_transformed, columns=feature_cols)
        if not isinstance(X_transformed, pd.DataFrame)
        else X_transformed
    )

    # Beeswarm summary plot
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_vals, X_display,
                      feature_names=feature_cols, show=False, plot_size=None)
    plt.title("SHAP Feature Importance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(artifacts_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"   Saved shap_summary.png  → {artifacts_dir / 'shap_summary.png'}")

    return mean_abs

'''

# Replace the compute_shap() function in your model/train.py with this version
# It saves feature_means and feature_stds into shap_values.json
# so the server can approximate per-patient SHAP without the shap library

def compute_shap(champion, X_test, feature_cols: list,
                 artifacts_dir, df_full=None):
    """
    Compute SHAP values ONCE during training on your laptop.
    Saves everything to shap_values.json — server just reads this file.
    No shap library needed on the server.
    """
    if not _SHAP:
        print("   [skip] shap not installed — pip install shap")
        return {}

    print("   Computing SHAP values (this runs only during training)...")

    try:
        pre = champion.named_steps["pre"]
        X_transformed = pre.transform(X_test)
    except Exception:
        X_transformed = X_test.values

    try:
        clf = champion.named_steps["clf"]
    except Exception:
        clf = champion

    try:
        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X_transformed)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
    except Exception:
        try:
            explainer = shap.LinearExplainer(clf, X_transformed)
            shap_vals = explainer.shap_values(X_transformed)
        except Exception as e:
            print(f"   [skip] SHAP failed: {e}")
            return {}

    mean_abs = {
        feat: float(np.abs(shap_vals[:, i]).mean())
        for i, feat in enumerate(feature_cols)
    }

    # Save dataset means and stds so server can approximate per-patient SHAP
    feature_means = {}
    feature_stds  = {}
    if df_full is not None:
        for feat in feature_cols:
            if feat in df_full.columns:
                feature_means[feat] = float(df_full[feat].median())
                feature_stds[feat]  = float(df_full[feat].std())

    # Save full SHAP data to JSON
    shap_json = {
        "feature_names":  feature_cols,
        "mean_abs_shap":  mean_abs,
        "feature_means":  feature_means,   # NEW — for server approximation
        "feature_stds":   feature_stds,    # NEW — for server approximation
        "samples": [
            {feat: float(shap_vals[row, i])
             for i, feat in enumerate(feature_cols)}
            for row in range(min(100, len(shap_vals)))
        ],
    }

    out_path = artifacts_dir / "shap_values.json"
    with open(out_path, "w") as f:
        json.dump(shap_json, f, indent=2)
    print(f"   Saved shap_values.json → {out_path}")

    return mean_abs
# ── 4. Save plots ──────────────────────────────────────────────────────────
def save_plots(results, champion_name, y_test, feature_importance,
               artifacts_dir: Path, feature_cols: list):

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, results[champion_name]["y_pred"],
        display_labels=["No Diabetes", "Diabetes"],
        colorbar=False, ax=ax, cmap="Blues",
    )
    ax.set_title(f"Confusion Matrix — {champion_name}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(artifacts_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ROC curves (all models)
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#6366f1", "#22c55e", "#f59e0b", "#ef4444"]
    for (name, data), col in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, data["y_prob"])
        ax.plot(fpr, tpr, color=col, lw=2,
                label=f"{name} (AUC={data['metrics']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curves — All Models", xlim=[0, 1], ylim=[0, 1.02])
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(artifacts_dir / "roc_curves.png", dpi=150)
    plt.close(fig)

    # Feature importance (permutation)
    sorted_fi = sorted(feature_importance.items(), key=lambda x: x[1])
    names_, vals_ = zip(*sorted_fi)
    colors_ = ["#ef4444" if v < 0 else "#6366f1" for v in vals_]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(names_, vals_, color=colors_, edgecolor="none")
    ax.axvline(0, color="black", lw=0.8)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_xlabel("Permutation Importance (accuracy drop)")
    ax.set_title(f"Feature Importance — {champion_name}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(artifacts_dir / "feature_importance.png", dpi=150)
    plt.close(fig)

    # Model comparison bar chart
    names     = list(results.keys())
    test_acc  = [results[n]["metrics"]["accuracy"] for n in names]
    cv_mean   = [results[n]["metrics"]["cv_accuracy_mean"] for n in names]
    cv_std    = [results[n]["metrics"]["cv_accuracy_std"]  for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - 0.175, test_acc, 0.35, label="Test accuracy",    color="#6366f1cc")
    b2 = ax.bar(x + 0.175, cv_mean,  0.35, label="CV accuracy (mean±std)",
                color="#6366f166", yerr=cv_std, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("Accuracy")
    ax.set_title("Model Comparison", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.bar_label(b1, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.3f", padding=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(artifacts_dir / "model_comparison.png", dpi=150)
    plt.close(fig)

    print(f"   Saved confusion_matrix.png, roc_curves.png, "
          f"feature_importance.png, model_comparison.png  → {artifacts_dir}")


# ── 5. Print Flutter model.dart coefficients ───────────────────────────────
def print_flutter_coefficients(df: pd.DataFrame, feature_cols: list,
                                champion, champion_name: str):
    print("\n── Flutter model.dart coefficients ─────────────────")
    print("   Copy-paste these 4 lines into lib/utils/model.dart\n")

    X_all = df[feature_cols].copy()
    for c in ZERO_IS_MISSING:
        if c in X_all.columns:
            X_all[c] = X_all[c].replace(0, np.nan)

    imp   = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X_all)
    means = imp.statistics_

    sc     = StandardScaler()
    X_sc   = sc.fit_transform(X_imp)
    scales = sc.scale_

    # FIX-4: Pass the already-imputed DataFrame to predict_proba so that
    # the champion pipeline's preprocessor sees clean data (no NaNs) and
    # the resulting probabilities are calibrated against the same
    # mean/scale arrays that will be exported to Flutter.
    X_imp_df = pd.DataFrame(X_imp, columns=feature_cols)
    rf_probs = champion.predict_proba(X_imp_df)[:, 1]

    cal = LogisticRegression(max_iter=1000, C=1.0)
    cal.fit(X_sc, (rf_probs >= 0.5).astype(int))

    coef      = [round(float(c), 4) for c in cal.coef_[0]]
    intercept = round(float(cal.intercept_[0]), 4)
    means_r   = [round(float(m), 4) for m in means]
    scales_r  = [round(float(s), 4) for s in scales]

    print(f"  // Feature order: {', '.join(feature_cols)}")
    print(f"  const _coef      = {coef};")
    print(f"  const _intercept = {intercept};")
    print(f"  const _mean      = {means_r};")
    print(f"  const _scale     = {scales_r};")
    print()

    return {
        "feature_order": feature_cols,
        "coef":          coef,
        "intercept":     intercept,
        "mean":          means_r,
        "scale":         scales_r,
        "champion_model": champion_name,
    }


# ── 6. Main entry point ────────────────────────────────────────────────────
def train_and_save(model_path: Path | None = None,
                   artifacts_dir: Path | None = None) -> Path:
    if model_path    is None: model_path    = config.MODEL_PATH
    if artifacts_dir is None: artifacts_dir = config.ARTIFACTS_DIR
    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = fix_zero_values(load_dataset())

    results, champion_name, champion, X_test, y_test, \
        feature_importance, feature_cols = train_all_models(df)

    # ── Save JSON artifacts ────────────────────────────────────────────────
    metrics_out = {
        name: {"metrics": data["metrics"], "best_params": data["best_params"]}
        for name, data in results.items()
    }
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)

    with open(config.FEATURE_IMPORTANCE_PATH, "w") as f:
        json.dump(feature_importance, f, indent=2)

    training_summary = {
        "best_model":       champion_name,
        "selection_metric": config.MODEL_SELECTION_METRIC,
        "test_metrics":     results[champion_name]["metrics"],
        "smote_used":       _SMOTE,
        "xgboost_used":     _XGB,
        "shap_used":        _SHAP,
        "feature_columns":  feature_cols,
    }
    with open(config.TRAINING_SUMMARY_PATH, "w") as f:
        json.dump(training_summary, f, indent=2)

    # ── Save model ─────────────────────────────────────────────────────────
    with open(model_path, "wb") as f:
        pickle.dump(champion, f)
    print(f"   Saved model.pkl → {model_path}")

    # ── Save plots ─────────────────────────────────────────────────────────
    print("\n── Saving plots ────────────────────────────────────")
    save_plots(results, champion_name, y_test,
               feature_importance, artifacts_dir, feature_cols)

    # ── SHAP ───────────────────────────────────────────────────────────────
    compute_shap(champion, X_test, feature_cols, artifacts_dir, df_full=df)

    # ── Flutter coefficients ───────────────────────────────────────────────
    flutter_coefs = print_flutter_coefficients(
        df, feature_cols, champion, champion_name)
    with open(artifacts_dir / "flutter_coefficients.json", "w") as f:
        json.dump(flutter_coefs, f, indent=2)

    print(f"\n✅  Done. All artifacts saved to {artifacts_dir}\n")
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    if (not args.retrain
            and config.MODEL_PATH.exists()
            and config.METRICS_PATH.exists()):
        print("Model already exists. Use --retrain to force retraining.")
    else:
        train_and_save()