#!/usr/bin/env python3
"""
train_isBuried_classifier.py

Train a logistic-regression classifier that predicts `buriedInside`
(1 = the domain is buried within its parent protein, 0 = it isn't)
from three structural metrics:

    anchoringIndex     (from AlphaFold PAE)
    fractionBuried     (from freeSASA: full vs isolated domain)
    contactDensity     (from BioPython: domain–nondomain contacts)

The model is intended to gate step 5: domains predicted as `1` are
filtered out of the library because their inter-domain interactions are
strong enough that they're likely not stand-alone folding units.

Design choices
--------------
- Pipeline = StandardScaler → LogisticRegression(L2, class_weight=balanced).
  Standardisation is INSIDE the pipeline so it can't leak from train→test.
- Hyperparameter search over `C` via 5-fold stratified CV on the training
  split (small dataset → CV reduces variance vs. a single holdout).
- 70/30 stratified train/test split for the headline holdout score, with
  CV-based performance estimate as the more reliable number (small N).
- Diagnostics: ROC, PR, calibration, confusion matrix, decision-threshold
  sweep, permutation feature importance, and a CV-fold boxplot. All go
  into --report-dir.
- The trained pipeline and the original feature list are pickled to
  --model-path so a separate prediction step (predict_isBuried.py or
  manual joblib.load) can score new step-3 outputs.

USAGE
-----
    python train_isBuried_classifier.py \\
        --truth-tsv  truthset_with_metrics.tsv \\
        --model-path isBuried_logreg.joblib \\
        --report-dir isBuried_training_report

Optional:
    --test-size 0.30        # holdout fraction (default 0.30)
    --random-state 42       # reproducibility seed (default 42)
    --predict-on TSV        # score this file with the trained model;
                            # writes <stem>_isBuried_pred.tsv next to it
"""

from __future__ import annotations
import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (auc, average_precision_score, brier_score_loss,
                              classification_report, confusion_matrix,
                              precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                      cross_val_predict, cross_val_score,
                                      train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

FEATURES = ["anchoringIndex", "fractionBuried", "contactDensity"]
TARGET   = "buriedInside"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_and_clean(tsv_path: Path):
    print(f"Loading {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t")
    for col in FEATURES + [TARGET]:
        if col not in df.columns:
            sys.exit(f"ERROR: input missing required column '{col}'. "
                     f"Got: {list(df.columns)}")
    n_total = len(df)

    # Coerce features to numeric and drop rows missing any feature or the label.
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df_clean = df.dropna(subset=FEATURES + [TARGET]).copy()
    df_clean[TARGET] = df_clean[TARGET].astype(int)

    n_dropped = n_total - len(df_clean)
    if n_dropped:
        print(f"  Dropped {n_dropped:,} rows with missing features/label.")
    print(f"  Working set: {len(df_clean):,} rows.")

    class_counts = df_clean[TARGET].value_counts().sort_index()
    print(f"  Class balance: "
          f"{class_counts.get(0, 0)} not-buried (0), "
          f"{class_counts.get(1, 0)} buried (1)")

    if df_clean[TARGET].nunique() < 2:
        sys.exit("ERROR: only one class present after cleaning — can't train.")
    if len(df_clean) < 20:
        print(f"  WARNING: only {len(df_clean)} usable rows. "
              "Model performance will be noisy; treat CV stddev as the "
              "headline uncertainty.")
    return df, df_clean


# ---------------------------------------------------------------------------
# EDA plots
# ---------------------------------------------------------------------------

def plot_feature_distributions(df, outdir):
    """Violin per feature, split by class."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, feat in zip(axes, FEATURES):
        data = [df.loc[df[TARGET] == c, feat].values for c in (0, 1)]
        parts = ax.violinplot(data, showmeans=False, showmedians=True,
                              positions=[0, 1], widths=0.7)
        for body, c in zip(parts["bodies"], ("#888888", "#C44E52")):
            body.set_facecolor(c); body.set_alpha(0.55)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["not buried (0)", "buried (1)"])
        ax.set_title(feat); ax.set_ylabel(feat)
        # Optional Cohen's d
        a, b = data
        if len(a) > 1 and len(b) > 1:
            s = np.sqrt(((len(a)-1)*np.var(a, ddof=1)
                          + (len(b)-1)*np.var(b, ddof=1))
                         / (len(a)+len(b)-2))
            if s > 0:
                d = (np.mean(b) - np.mean(a)) / s
                ax.text(0.02, 0.97, f"d = {d:+.2f}",
                        transform=ax.transAxes, va="top", fontsize=9,
                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    fig.suptitle("Feature distributions by class", fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / "01_feature_distributions.png",
                dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_pair_grid(df, outdir):
    """Pairwise scatter of features coloured by class."""
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    colors = df[TARGET].map({0: "#888888", 1: "#C44E52"})
    for i, fi in enumerate(FEATURES):
        for j, fj in enumerate(FEATURES):
            ax = axes[i, j]
            if i == j:
                for c in (0, 1):
                    vals = df.loc[df[TARGET] == c, fi].values
                    ax.hist(vals, bins=15, alpha=0.5,
                            color="#888888" if c == 0 else "#C44E52",
                            label=str(c))
                ax.set_title(fi, fontsize=9)
                if i == 0: ax.legend(fontsize=8, title=TARGET)
            else:
                ax.scatter(df[fj], df[fi], c=colors, s=18, alpha=0.7,
                           edgecolor="white", linewidth=0.4)
            if i == 2: ax.set_xlabel(fj, fontsize=9)
            if j == 0: ax.set_ylabel(fi, fontsize=9)
            ax.tick_params(labelsize=7)
    fig.suptitle("Feature pairwise scatter (red = buried, grey = not)",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(outdir / "02_pairwise_scatter.png",
                dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_correlation_heatmap(df, outdir):
    """Pearson correlation between features (note: features are correlated
    by construction — anchoringIndex, fractionBuried, contactDensity all
    measure 'parent-domain coupling'). High correlation → coefficients
    will be hard to interpret individually; use permutation importance for
    a model-agnostic view."""
    corr = df[FEATURES].corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(FEATURES))); ax.set_xticklabels(FEATURES, rotation=45, ha="right")
    ax.set_yticks(range(len(FEATURES))); ax.set_yticklabels(FEATURES)
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            ax.text(j, i, f"{corr.iat[i,j]:.2f}", ha="center", va="center",
                    color="white" if abs(corr.iat[i,j]) > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Feature Pearson correlation")
    fig.tight_layout()
    fig.savefig(outdir / "03_correlation_heatmap.png",
                dpi=150, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
# Model evaluation plots
# ---------------------------------------------------------------------------

def plot_roc_pr_calibration(y_true, y_score, outdir):
    auc_score = roc_auc_score(y_true, y_score)
    ap_score  = average_precision_score(y_true, y_score)
    brier     = brier_score_loss(y_true, y_score)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    # ROC
    axes[0].plot(fpr, tpr, color="#C44E52",
                 label=f"AUC = {auc_score:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", alpha=0.6)
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve (test set)"); axes[0].legend(loc="lower right")
    # PR
    base = y_true.mean()
    axes[1].plot(rec, prec, color="#4C72B0",
                 label=f"AP = {ap_score:.3f}")
    axes[1].axhline(base, ls="--", color="grey", alpha=0.6,
                    label=f"baseline = {base:.2f}")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision–Recall (test)"); axes[1].legend(loc="lower left")
    # Calibration
    n_bins = max(3, min(8, len(y_true) // 5))
    frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=n_bins,
                                             strategy="quantile")
    axes[2].plot(mean_pred, frac_pos, "o-", color="#55A868")
    axes[2].plot([0, 1], [0, 1], "--", color="grey", alpha=0.6)
    axes[2].set_xlabel("Mean predicted probability")
    axes[2].set_ylabel("Fraction of positives")
    axes[2].set_title(f"Calibration  (Brier = {brier:.3f})")

    fig.tight_layout()
    fig.savefig(outdir / "04_roc_pr_calibration.png",
                dpi=150, bbox_inches="tight"); plt.close(fig)
    return auc_score, ap_score, brier


def plot_confusion_matrix(y_true, y_pred, outdir, fname, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["not buried", "buried"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["not buried", "buried"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(outdir / fname, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_threshold_sweep(y_true, y_score, outdir):
    """Precision/recall/F1 vs decision threshold. Marks F1-optimal threshold."""
    thresholds = np.linspace(0.05, 0.95, 91)
    prec, rec, f1 = [], [], []
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        f = 2*p*r/(p+r) if (p+r) else 0
        prec.append(p); rec.append(r); f1.append(f)
    prec, rec, f1 = map(np.array, (prec, rec, f1))
    t_best = thresholds[int(f1.argmax())]
    f1_best = f1.max()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, prec, label="precision", color="#4C72B0")
    ax.plot(thresholds, rec,  label="recall",    color="#DD8452")
    ax.plot(thresholds, f1,   label="F1",        color="#55A868")
    ax.axvline(0.5, ls=":", color="grey", alpha=0.5, label="default 0.5")
    ax.axvline(t_best, ls="--", color="#55A868", alpha=0.7,
               label=f"F1-best ({t_best:.2f} → F1={f1_best:.2f})")
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Score")
    ax.set_title("Threshold sweep on test set")
    ax.legend(loc="lower center", ncol=2); ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(outdir / "06_threshold_sweep.png",
                dpi=150, bbox_inches="tight"); plt.close(fig)
    return float(t_best), float(f1_best)


def plot_cv_scores(scores_dict, outdir):
    """Boxplot of per-fold scores across metrics."""
    names = list(scores_dict.keys())
    vals  = [scores_dict[n] for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(vals, labels=names, showmeans=True)
    ax.set_title("Cross-validation per-fold scores (5-fold)")
    ax.set_ylabel("Score"); ax.set_ylim(-0.02, 1.02)
    for i, v in enumerate(vals):
        ax.scatter([i+1]*len(v), v, color="#C44E52", alpha=0.55, s=18, zorder=3)
    fig.tight_layout()
    fig.savefig(outdir / "07_cv_scores.png",
                dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_feature_importance(model, X_test, y_test, outdir, random_state):
    """Two views of importance: standardised coefficients + permutation."""
    lr = model.named_steps["lr"]
    coef = lr.coef_[0]
    perm = permutation_importance(model, X_test, y_test,
                                  n_repeats=30, random_state=random_state,
                                  scoring="roc_auc")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    pos = np.arange(len(FEATURES))
    colors = ["#C44E52" if c > 0 else "#4C72B0" for c in coef]
    axes[0].barh(pos, coef, color=colors)
    axes[0].set_yticks(pos); axes[0].set_yticklabels(FEATURES)
    axes[0].axvline(0, color="black", lw=0.5)
    axes[0].set_xlabel("Standardised LR coefficient")
    axes[0].set_title("Direction & magnitude\n(positive → pushes 'buried')")

    axes[1].barh(pos, perm.importances_mean,
                 xerr=perm.importances_std, color="#888888")
    axes[1].set_yticks(pos); axes[1].set_yticklabels(FEATURES)
    axes[1].set_xlabel("Permutation importance (ΔROC-AUC on test)")
    axes[1].set_title("Permutation importance (model-agnostic)")
    fig.tight_layout()
    fig.savefig(outdir / "08_feature_importance.png",
                dpi=150, bbox_inches="tight"); plt.close(fig)
    return coef, perm.importances_mean, perm.importances_std


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth-tsv", required=True, type=Path,
                    help="Truth-set TSV with the 3 features + buriedInside column.")
    ap.add_argument("--model-path", required=True, type=Path,
                    help="Where to save the trained pipeline (joblib).")
    ap.add_argument("--report-dir", required=True, type=Path,
                    help="Directory to write evaluation plots + summary JSON.")
    ap.add_argument("--test-size", type=float, default=0.30,
                    help="Stratified holdout fraction (default 0.30).")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--predict-on", type=Path, default=None,
                    help="Optional: score this TSV with the trained model. "
                         "Writes a sibling file with isBuried_prob and "
                         "isBuried_pred columns appended.")
    args = ap.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load + clean
    df_raw, df = load_and_clean(args.truth_tsv)

    # 2. EDA plots
    print("\nEDA plots ...")
    plot_feature_distributions(df, args.report_dir)
    plot_pair_grid(df, args.report_dir)
    plot_correlation_heatmap(df, args.report_dir)

    # 3. Train/test split (stratified)
    X = df[FEATURES].values
    y = df[TARGET].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, stratify=y,
        random_state=args.random_state)
    print(f"\nSplit: {len(y_tr)} train / {len(y_te)} test "
          f"(stratified, random_state={args.random_state})")
    print(f"  Train class balance: 0={int((y_tr==0).sum())}, 1={int((y_tr==1).sum())}")
    print(f"  Test  class balance: 0={int((y_te==0).sum())}, 1={int((y_te==1).sum())}")

    # 4. Pipeline (scaler + LR). Class weight balanced for slight imbalance.
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(
                       penalty="l2", solver="lbfgs",
                       class_weight="balanced",
                       max_iter=5000)),
    ])

    # 5. Grid-search C via 5-fold CV on the training data
    cv = StratifiedKFold(n_splits=5, shuffle=True,
                         random_state=args.random_state)
    grid = GridSearchCV(pipe, {"lr__C": [0.01, 0.1, 1, 3, 10, 30, 100]},
                        scoring="roc_auc", cv=cv, n_jobs=1)
    grid.fit(X_tr, y_tr)
    best_C = grid.best_params_["lr__C"]
    print(f"\nGrid-search best C = {best_C}  "
          f"(mean train-CV ROC-AUC = {grid.best_score_:.3f})")
    model = grid.best_estimator_

    # 6. Cross-validated scores on the FULL training set with best C
    print("\n5-fold CV on training set (with best C):")
    cv_scores = {}
    for metric in ("roc_auc", "average_precision", "f1", "accuracy"):
        s = cross_val_score(model, X_tr, y_tr, cv=cv, scoring=metric,
                            n_jobs=1)
        cv_scores[metric] = s
        print(f"  {metric:18s}: {s.mean():.3f} ± {s.std():.3f}  "
              f"(folds: {[round(v,2) for v in s]})")
    plot_cv_scores(cv_scores, args.report_dir)

    # 7. Test-set evaluation
    y_score = model.predict_proba(X_te)[:, 1]
    y_pred  = (y_score >= 0.5).astype(int)
    print("\nTest-set classification report (threshold = 0.50):")
    print(classification_report(y_te, y_pred, target_names=["not_buried","buried"]))
    auc_t, ap_t, brier_t = plot_roc_pr_calibration(y_te, y_score, args.report_dir)
    plot_confusion_matrix(y_te, y_pred, args.report_dir,
                          "05_confusion_matrix_test.png",
                          "Confusion matrix (test, threshold = 0.50)")
    t_best, f1_best = plot_threshold_sweep(y_te, y_score, args.report_dir)

    # Confusion matrix at F1-optimal threshold too
    y_pred_opt = (y_score >= t_best).astype(int)
    plot_confusion_matrix(y_te, y_pred_opt, args.report_dir,
                          "05b_confusion_matrix_test_optThr.png",
                          f"Confusion matrix (test, threshold = {t_best:.2f}, F1 = {f1_best:.2f})")

    # 8. Feature importance
    coef, perm_mean, perm_std = plot_feature_importance(
        model, X_te, y_te, args.report_dir, args.random_state)

    # 9. Save model + scaler in a single bundle, plus a metadata JSON
    bundle = {
        "model": model,                       # whole sklearn pipeline
        "features": FEATURES,
        "target": TARGET,
        "best_C": best_C,
        "thresholds": {"default": 0.5, "f1_optimal": t_best},
    }
    joblib.dump(bundle, args.model_path)
    print(f"\nSaved model bundle → {args.model_path}")

    summary = {
        "n_train": int(len(y_tr)),
        "n_test":  int(len(y_te)),
        "best_C":  float(best_C),
        "cv_roc_auc_mean": float(cv_scores["roc_auc"].mean()),
        "cv_roc_auc_std":  float(cv_scores["roc_auc"].std()),
        "cv_f1_mean":      float(cv_scores["f1"].mean()),
        "cv_f1_std":       float(cv_scores["f1"].std()),
        "test_roc_auc":  float(auc_t),
        "test_avg_prec": float(ap_t),
        "test_brier":    float(brier_t),
        "f1_optimal_threshold": float(t_best),
        "f1_optimal_test_score": float(f1_best),
        "feature_coefs": dict(zip(FEATURES, [float(c) for c in coef])),
        "permutation_importance_mean":
            dict(zip(FEATURES, [float(v) for v in perm_mean])),
        "permutation_importance_std":
            dict(zip(FEATURES, [float(v) for v in perm_std])),
    }
    (args.report_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved metrics summary  → {args.report_dir / 'summary.json'}")

    # 10. Optional prediction pass on a separate TSV
    if args.predict_on:
        target = args.predict_on
        print(f"\nScoring {target} with the trained model ...")
        new_df = pd.read_csv(target, sep="\t")
        for c in FEATURES:
            if c not in new_df.columns:
                sys.exit(f"ERROR: --predict-on file missing column '{c}'.")
        Xnew = new_df[FEATURES].apply(pd.to_numeric, errors="coerce").values
        valid_mask = ~np.isnan(Xnew).any(axis=1)
        probs = np.full(len(new_df), np.nan)
        probs[valid_mask] = model.predict_proba(Xnew[valid_mask])[:, 1]
        preds_default = (probs >= 0.5).astype(float)
        preds_opt     = (probs >= t_best).astype(float)
        # NaN-aware: when probability is NaN, prediction is NaN
        preds_default[np.isnan(probs)] = np.nan
        preds_opt[np.isnan(probs)] = np.nan
        new_df["isBuried_prob"]        = probs
        new_df["isBuried_pred_t050"]   = preds_default
        new_df["isBuried_pred_tBest"]  = preds_opt
        out_path = target.with_name(target.stem + "_isBuried_pred" + target.suffix)
        new_df.to_csv(out_path, sep="\t", index=False)
        print(f"  Wrote {out_path}")
        print(f"  Rows scored:     {int(valid_mask.sum())}/{len(new_df)}")
        print(f"  Rows skipped (NaN in features): {int((~valid_mask).sum())}")
        print(f"  Predicted buried (t=0.50):     {int(np.nansum(preds_default))}")
        print(f"  Predicted buried (t={t_best:.2f}):  {int(np.nansum(preds_opt))}")


if __name__ == "__main__":
    main()
