"""Generate LaTeX table fragments from the results/ CSVs for inclusion in the report."""
import os
import re
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
TABLES_DIR = os.path.join(BASE, "report", "tables")
LISTINGS_DIR = os.path.join(BASE, "report", "listings")
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(LISTINGS_DIR, exist_ok=True)

DATASETS = ["auto_mpg", "concrete", "airfoil"]


def escape(s):
    return str(s).replace("_", "\\_").replace("%", "\\%")


def summary_table(name):
    df = pd.read_csv(os.path.join(RESULTS_DIR, name, "summary_stats.csv"), index_col=0)
    lines = [r"\begin{tabular}{l" + "r" * len(df.columns) + "}",
             r"\toprule",
             "Feature & " + " & ".join(escape(c) for c in df.columns) + r" \\",
             r"\midrule"]
    for idx, row in df.iterrows():
        vals = " & ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row)
        lines.append(f"{escape(idx)} & {vals} " + r"\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    with open(os.path.join(TABLES_DIR, f"{name}_summary.tex"), "w") as f:
        f.write("\n".join(lines))


def corr_table(name, target):
    corr = pd.read_csv(os.path.join(RESULTS_DIR, name, "corr_matrix.csv"), index_col=0)
    s = corr[target].drop(target).sort_values(key=lambda x: -x.abs())
    lines = [r"\begin{tabular}{lr}",
             r"\toprule",
             r"Feature & Correlation with target \\",
             r"\midrule"]
    for idx, v in s.items():
        lines.append(f"{escape(idx)} & {v:.4f} " + r"\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    with open(os.path.join(TABLES_DIR, f"{name}_target_corr.tex"), "w") as f:
        f.write("\n".join(lines))


def full_corr_table(name):
    corr = pd.read_csv(os.path.join(RESULTS_DIR, name, "corr_matrix.csv"), index_col=0)
    cols = list(corr.columns)
    # abbreviate long column names for the header row so the table fits the page,
    # disambiguating any collisions (e.g. fly_ash / fine_aggregate both -> "FA")
    abbrev, seen = {}, set()
    for c in cols:
        base = c if len(c) <= 4 else "".join(w[0] for w in c.split("_")).upper()
        candidate, n = base, 1
        while candidate in seen:
            n += 1
            candidate = f"{base}{n}"
        seen.add(candidate)
        abbrev[c] = candidate
    lines = [r"\begin{tabular}{l" + "r" * len(cols) + "}",
             r"\toprule",
             "Feature & " + " & ".join(escape(abbrev[c]) for c in cols) + r" \\",
             r"\midrule"]
    for idx, row in corr.iterrows():
        vals = " & ".join(f"{v:.2f}" for v in row)
        lines.append(f"{escape(idx)} & {vals} " + r"\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    legend = "; ".join(f"{escape(abbrev[c])}={escape(c)}" for c in cols if abbrev[c] != c)
    with open(os.path.join(TABLES_DIR, f"{name}_full_corr.tex"), "w") as f:
        f.write("\n".join(lines))
    with open(os.path.join(TABLES_DIR, f"{name}_full_corr_legend.tex"), "w") as f:
        f.write(legend)


def extract_scalation_reports(name, features):
    """Pull the REPORT+SUMMARY block for each feature out of the raw sbt run log,
    stripping the noisy load/write lines, for verbatim inclusion in the report."""
    with open(os.path.join(RESULTS_DIR, name, "scalation_output.txt"),
              encoding="utf-8-sig") as f:
        text = f.read()
    blocks = re.split(r"\nb = VectorD", text)[1:]  # each starts right after "b = VectorD"
    for feat, block in zip(features, blocks):
        block = "b = VectorD" + block
        # cut off anything after the SUMMARY table's closing rule (2nd dashed line after "SUMMARY")
        m = re.search(r"(REPORT.*?SUMMARY.*?\n-{30,}\n.*?\n-{30,}\n)", block, re.DOTALL)
        clean = m.group(1) if m else block
        clean = clean.replace("\r\n", "\n").strip("\n")
        with open(os.path.join(LISTINGS_DIR, f"{name}_{feat}_scalation.txt"), "w") as f:
            f.write(clean)


TARGETS = {"auto_mpg": "mpg", "concrete": "concrete_compressive_strength",
           "airfoil": "scaled_sound_pressure_level"}
TOP_FEATURES = {"auto_mpg": ["weight", "displacement"],
                "concrete": ["cement", "superplasticizer"],
                "airfoil": ["frequency", "suction_side_displacement_thickness"]}

for name in DATASETS:
    summary_table(name)
    corr_table(name, TARGETS[name])
    full_corr_table(name)
    extract_scalation_reports(name, TOP_FEATURES[name])

print("Tables written to", TABLES_DIR)
print("Listings written to", LISTINGS_DIR)
