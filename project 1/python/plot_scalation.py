"""Render ScalaTion's (x, y, yp) prediction CSVs as fit-vs-actual plots,
mirroring the Statsmodels plots produced by eda_regression.py, so both
tools' results are visualized side by side in the report."""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")

for pred_csv in glob.glob(os.path.join(RESULTS_DIR, "*", "scalation_pred_*.csv")):
    dataset_dir = os.path.basename(os.path.dirname(pred_csv))
    feat = os.path.basename(pred_csv)[len("scalation_pred_"):-len(".csv")]
    df = pd.read_csv(pred_csv)
    x_col, y_col = df.columns[0], df.columns[1]
    x, y, yp = df[x_col].values, df[y_col].values, df["yp"].values
    order = np.argsort(x)

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, color="black", s=15, label="actual (y)")
    plt.plot(x[order], yp[order], color="red", linewidth=2, label="predicted (y-hat)")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{dataset_dir}: {y_col} vs {x_col} (ScalaTion)")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, dataset_dir, f"scalation_fit_vs_actual_{feat}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("wrote", out_path)
