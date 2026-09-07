"""
Project 1: Exploratory Data Analysis (EDA) and Simple Regression
Runs EDA + statsmodels Simple (univariate OLS) Regression on the top two
features (by |correlation| with the target) for each of the three UCI
datasets: Auto MPG, Concrete Compressive Strength, Airfoil Self-Noise.

Outputs (per dataset) are written to project 1/results/<dataset>/:
  - summary_stats.csv        descriptive statistics for every feature
  - corr_matrix.csv          full correlation matrix
  - corr_heatmap.png         heatmap of the correlation matrix
  - ols_summary_<feat>.txt   statsmodels OLS fit report for y ~ feat
  - fit_vs_actual_<feat>.png plot of y and y-hat vs. x for feat
  - top_features.txt         the chosen top-2 features and their correlations
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
RESULTS_DIR = os.path.join(BASE, "results")


def prep_concrete_csv():
    """Concrete data ships as .xls; convert once to a clean numeric .csv."""
    xls_path = os.path.join(DATA_DIR, "raw", "Concrete_Data.xls")
    csv_path = os.path.join(DATA_DIR, "concrete.csv")
    df = pd.read_excel(xls_path)
    cols = ["cement", "blast_furnace_slag", "fly_ash", "water",
             "superplasticizer", "coarse_aggregate", "fine_aggregate",
             "age", "concrete_compressive_strength"]
    df.columns = cols
    df.to_csv(csv_path, index=False)
    return csv_path


DATASETS = [
    {
        "name": "auto_mpg",
        "csv": os.path.join(DATA_DIR, "auto_mpg.csv"),
        "target": "mpg",
        "expected_rows": 392,
        "expected_cols": 8,
    },
    {
        "name": "concrete",
        "csv": None,  # filled in after prep_concrete_csv()
        "target": "concrete_compressive_strength",
        "expected_rows": 1030,
        "expected_cols": 9,
    },
    {
        "name": "airfoil",
        "csv": os.path.join(DATA_DIR, "airfoil.csv"),
        "target": "scaled_sound_pressure_level",
        "expected_rows": 1503,
        "expected_cols": 6,
    },
]


def preprocess(df, name):
    """Handle missing values and outliers; return cleaned df + a text log."""
    log = []
    n0 = len(df)

    missing = df.isna().sum()
    n_missing = int(missing.sum())
    if n_missing > 0:
        df = df.dropna()
        log.append(f"Dropped {n0 - len(df)} rows with missing values "
                    f"(columns affected: {list(missing[missing > 0].index)}).")
    else:
        log.append("No missing values found (dataset was pre-cleaned during "
                    "the .data/.dat -> .csv conversion for auto_mpg/airfoil, "
                    "or had none originally for concrete).")

    # string / non-numeric columns
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        log.append(f"Dropped non-numeric columns not used in regression: {non_numeric}.")
        df = df.drop(columns=non_numeric)
    else:
        log.append("No string columns present (car name / identifier columns, "
                    "if any, were excluded during CSV conversion).")

    # outlier detection via IQR rule, reported but NOT removed
    # (physically valid extreme values, e.g. high-horsepower cars or
    # low-age concrete, are kept; removing them would discard real
    # information for a dataset this small).
    outlier_report = {}
    for col in df.columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
        if n_out > 0:
            outlier_report[col] = n_out
    if outlier_report:
        log.append(f"IQR-rule outlier counts per column (flagged, not removed, "
                    f"since they reflect real variability in a modest-size "
                    f"dataset): {outlier_report}.")
    else:
        log.append("No IQR-rule outliers detected in any column.")

    return df, log


def analyze(cfg):
    name, target = cfg["name"], cfg["target"]
    out_dir = os.path.join(RESULTS_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    df_raw = pd.read_csv(cfg["csv"])
    df, prep_log = preprocess(df_raw, name)

    with open(os.path.join(out_dir, "preprocessing_log.txt"), "w") as f:
        f.write(f"Preprocessing log for {name}\n")
        f.write(f"Raw shape: {df_raw.shape}, cleaned shape: {df.shape}\n")
        f.write(f"Expected: {cfg['expected_rows']} rows, {cfg['expected_cols']} columns\n\n")
        for line in prep_log:
            f.write(f"- {line}\n")

    # 1. statistical summaries
    summary = df.describe().T
    summary.to_csv(os.path.join(out_dir, "summary_stats.csv"))

    # 2. correlation matrix + heatmap
    corr = df.corr()
    corr.to_csv(os.path.join(out_dir, "corr_matrix.csv"))

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": 0.8})
    plt.title(f"Correlation Heatmap: {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "corr_heatmap.png"), dpi=150)
    plt.close()

    # 3. top two features by |correlation| with target
    target_corr = corr[target].drop(target).abs().sort_values(ascending=False)
    top2 = target_corr.index[:2].tolist()

    with open(os.path.join(out_dir, "top_features.txt"), "w") as f:
        f.write(f"Target: {target}\n")
        f.write("Correlation with target (sorted by |r|):\n")
        for feat, r in corr[target].drop(target).abs().sort_values(ascending=False).items():
            signed = corr[target][feat]
            f.write(f"  {feat}: r = {signed:.4f}\n")
        f.write(f"\nTop 2 features selected for simple regression: {top2}\n")

    # 4. simple (univariate) OLS regression for each top feature
    results = {}
    for feat in top2:
        x = df[feat].values
        y = df[target].values
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        with open(os.path.join(out_dir, f"ols_summary_{feat}.txt"), "w") as f:
            f.write(str(model.summary()))

        yhat = model.predict(X)
        order = np.argsort(x)
        plt.figure(figsize=(7, 5))
        plt.scatter(x, y, color="black", s=15, label="actual (y)")
        plt.plot(x[order], yhat[order], color="red", linewidth=2, label="predicted (y-hat)")
        plt.xlabel(feat)
        plt.ylabel(target)
        plt.title(f"{name}: {target} vs {feat} (Statsmodels OLS)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"fit_vs_actual_{feat}.png"), dpi=150)
        plt.close()

        results[feat] = {
            "b0": model.params[0], "b1": model.params[1],
            "rsq": model.rsquared, "rsq_adj": model.rsquared_adj,
            "fstat": model.fvalue, "pvalue": model.f_pvalue,
            "aic": model.aic, "bic": model.bic,
        }

    return {"name": name, "shape": df.shape, "top2": top2, "results": results,
            "target_corr": corr[target].drop(target)}


def main():
    concrete_cfg = next(c for c in DATASETS if c["name"] == "concrete")
    concrete_cfg["csv"] = prep_concrete_csv()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_all = []
    for cfg in DATASETS:
        print(f"=== {cfg['name']} ===")
        res = analyze(cfg)
        summary_all.append(res)
        print(f"  cleaned shape: {res['shape']}")
        print(f"  top 2 features: {res['top2']}")
        for feat, r in res["results"].items():
            print(f"    {feat}: R^2={r['rsq']:.4f}, b0={r['b0']:.4f}, b1={r['b1']:.4f}")

    with open(os.path.join(RESULTS_DIR, "all_results_summary.txt"), "w") as f:
        for res in summary_all:
            f.write(f"=== {res['name']} ===\n")
            f.write(f"cleaned shape: {res['shape']}\n")
            f.write(f"top 2 features: {res['top2']}\n")
            for feat, r in res["results"].items():
                f.write(f"  {feat}: R^2={r['rsq']:.4f}, adjR^2={r['rsq_adj']:.4f}, "
                        f"b0={r['b0']:.4f}, b1={r['b1']:.4f}, "
                        f"F={r['fstat']:.2f}, p={r['pvalue']:.3e}\n")
            f.write("\n")

    print("\nAll results written to", RESULTS_DIR)


if __name__ == "__main__":
    main()
