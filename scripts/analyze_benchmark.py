"""
Analyze benchmark summary tables and generate stats/plots.
Inputs are hardcoded from the provided per-dataset summaries.
Outputs:
- figures/benchmark_mean_accuracy.png (mean ± min/max bars)
- figures/benchmark_diff_range.png (spread bars)
- figures/benchmark_per_dataset.png (per-dataset accuracies)
- figures/benchmark_confidence_intervals.png (95% CI across datasets)
- results/benchmark_stats_summary.csv (aggregate stats)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent.parent
FIG_DIR = HERE / "figures"
RESULTS_DIR = HERE / "results"


def load_data() -> pd.DataFrame:
    # Per-dataset mean/min/max summaries (from benchmark runs)
    data: Dict[str, List[Dict]] = {
        "MNIST": [
            {"model": "digital", "acc_mean": 0.976, "acc_min": 0.976, "acc_max": 0.976, "diff": 0.0},
            {"model": "amplitude", "acc_mean": 0.97297, "acc_min": 0.9668, "acc_max": 0.9749, "diff": 0.0081},
            {"model": "amplitude_noiseaware", "acc_mean": 0.97214, "acc_min": 0.971, "acc_max": 0.9726, "diff": 0.0016},
            {"model": "phase", "acc_mean": 0.78415, "acc_min": 0.2754, "acc_max": 0.9787, "diff": 0.7033},
            {"model": "phase_noiseaware", "acc_mean": 0.94876, "acc_min": 0.802, "acc_max": 0.9737, "diff": 0.1717},
        ],
        "KMNIST": [
            {"model": "digital", "acc_mean": 0.8913, "acc_min": 0.8913, "acc_max": 0.8913, "diff": 0.0},
            {"model": "amplitude", "acc_mean": 0.87966, "acc_min": 0.8733, "acc_max": 0.882, "diff": 0.0087},
            {"model": "amplitude_noiseaware", "acc_mean": 0.88777, "acc_min": 0.8846, "acc_max": 0.8901, "diff": 0.0055},
            {"model": "phase", "acc_mean": 0.7454, "acc_min": 0.3103, "acc_max": 0.8958, "diff": 0.5855},
            {"model": "phase_noiseaware", "acc_mean": 0.85296, "acc_min": 0.7051, "acc_max": 0.8832, "diff": 0.1781},
        ],
        "EMNIST": [
            {"model": "digital", "acc_mean": 0.903317308, "acc_min": 0.903317308, "acc_max": 0.903317308, "diff": 0.0},
            {"model": "amplitude", "acc_mean": 0.900913462, "acc_min": 0.873413462, "acc_max": 0.907259615, "diff": 0.033846154},
            {"model": "amplitude_noiseaware", "acc_mean": 0.898533654, "acc_min": 0.895625, "acc_max": 0.899903846, "diff": 0.004278846},
            {"model": "phase", "acc_mean": 0.568951923, "acc_min": 0.122836538, "acc_max": 0.910913462, "diff": 0.788076923},
            {"model": "phase_noiseaware", "acc_mean": 0.892322115, "acc_min": 0.855, "acc_max": 0.9025, "diff": 0.0475},
        ],
        "CIFAR10": [
            {"model": "digital", "acc_mean": 0.5189, "acc_min": 0.5189, "acc_max": 0.5189, "diff": 0.0},
            {"model": "amplitude", "acc_mean": 0.51489, "acc_min": 0.4967, "acc_max": 0.5217, "diff": 0.025},
            {"model": "amplitude_noiseaware", "acc_mean": 0.52177, "acc_min": 0.5208, "acc_max": 0.5226, "diff": 0.0018},
            {"model": "phase", "acc_mean": 0.31185, "acc_min": 0.1246, "acc_max": 0.5257, "diff": 0.4011},
            {"model": "phase_noiseaware", "acc_mean": 0.45151, "acc_min": 0.4057, "acc_max": 0.4647, "diff": 0.059},
        ],
        "FMNIST": [
            {"model": "digital", "acc_mean": 0.8899, "acc_min": 0.8899, "acc_max": 0.8899, "diff": 0.0},
            {"model": "amplitude", "acc_mean": 0.88249, "acc_min": 0.8687, "acc_max": 0.8855, "diff": 0.0168},
            {"model": "amplitude_noiseaware", "acc_mean": 0.88441, "acc_min": 0.8829, "acc_max": 0.8853, "diff": 0.0024},
            {"model": "phase", "acc_mean": 0.66751, "acc_min": 0.2295, "acc_max": 0.8919, "diff": 0.6624},
            {"model": "phase_noiseaware", "acc_mean": 0.8553, "acc_min": 0.714, "acc_max": 0.8792, "diff": 0.1652},
        ],
    }
    records = []
    for dataset, entries in data.items():
        for row in entries:
            rec = {"dataset": dataset}
            rec.update(row)
            records.append(rec)
    return pd.DataFrame(records)


def aggregate_stats(df: pd.DataFrame) -> pd.DataFrame:
    # Mean over datasets for each model (matches provided averages)
    grouped = (
        df.groupby("model")
        .agg(
            acc_mean=("acc_mean", "mean"),
            acc_min=("acc_min", "mean"),
            acc_max=("acc_max", "mean"),
            diff=("diff", "mean"),
            acc_std=("acc_mean", "std"),
        )
        .reset_index()
    )
    # 95% CI across datasets (n=5)
    n = df["dataset"].nunique()
    tval = stats.t.ppf(0.975, df=n - 1)
    grouped["acc_ci_halfwidth"] = grouped["acc_std"] / (n ** 0.5) * tval
    # Delta vs digital
    digital_mean = grouped.loc[grouped["model"] == "digital", "acc_mean"].iloc[0]
    grouped["delta_vs_digital"] = grouped["acc_mean"] - digital_mean
    return grouped


def plot_mean_with_error(df: pd.DataFrame):
    FIG_DIR.mkdir(exist_ok=True)
    plt.style.use("seaborn-v0_8-colorblind")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df["model"], df["acc_mean"], color="#4B79A1")
    yerr = [df["acc_mean"] - df["acc_min"], df["acc_max"] - df["acc_mean"]]
    ax.errorbar(df["model"], df["acc_mean"], yerr=yerr, fmt="none", ecolor="black", capsize=5, lw=1)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Mean accuracy with min/max across datasets")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "benchmark_mean_accuracy.png", dpi=150)
    plt.close(fig)


def plot_spread(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["model"], df["diff"], color="#F4A261")
    ax.set_ylabel("Accuracy spread (max - min)")
    ax.set_title("Robustness spread across datasets/noise")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "benchmark_diff_range.png", dpi=150)
    plt.close(fig)


def plot_per_dataset(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    for model, sub in df.groupby("model"):
        ax.plot(sub["dataset"], sub["acc_mean"], marker="o", label=model)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-dataset mean accuracy by model")
    ax.set_ylim(0, 1.0)
    plt.xticks(rotation=20, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "benchmark_per_dataset.png", dpi=150)
    plt.close(fig)


def plot_conf_intervals(summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = range(len(summary))
    ax.errorbar(
        summary["acc_mean"],
        y,
        xerr=summary["acc_ci_halfwidth"],
        fmt="o",
        color="#2A9D8F",
        ecolor="black",
        capsize=4,
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(summary["model"])
    ax.set_xlabel("Accuracy (95% CI across datasets)")
    ax.set_title("Cross-dataset variability (approx. 95% CI)")
    ax.set_xlim(0, 1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "benchmark_confidence_intervals.png", dpi=150)
    plt.close(fig)


def save_summary(summary: pd.DataFrame):
    RESULTS_DIR.mkdir(exist_ok=True)
    summary.to_csv(RESULTS_DIR / "benchmark_stats_summary.csv", index=False)


def main() -> None:
    df = load_data()
    summary = aggregate_stats(df)
    plot_mean_with_error(summary)
    plot_spread(summary)
    plot_per_dataset(df)
    plot_conf_intervals(summary)
    save_summary(summary)
    print("Wrote figures to", FIG_DIR)
    print("Wrote stats to", RESULTS_DIR / "benchmark_stats_summary.csv")


if __name__ == "__main__":
    main()
