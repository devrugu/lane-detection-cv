"""Build the headline comparison figure for the report and presentation.

Combines:
  - Panel A: Accuracy / FPR / FNR table (text rendering)
  - Panel B: FPS bar chart
  - Panel C: Temporal smoothing jitter reduction

Outputs: results/summary_comparison.png (also saved as PDF for the report).
"""

from __future__ import annotations
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend


RESULTS_DIR = Path("results")


def load_metrics() -> dict:
    """Read all four input files and pull out the numbers we need."""
    out = {}

    # Classical TuSimple metrics
    with open(RESULTS_DIR / "tusimple_metrics_classical.json") as f:
        out["classical"] = json.load(f)

    # U-Net TuSimple metrics
    unet_path = RESULTS_DIR / "tusimple_metrics_unet.json"
    if unet_path.exists():
        with open(unet_path) as f:
            out["unet"] = json.load(f)
    else:
        # Hardcoded fallback from your run if file is missing
        out["unet"] = {
            "accuracy": 0.9953,
            "fpr": 0.2094,
            "fnr": 0.5897,
            "n_frames": 100,
        }
        print("WARNING: tusimple_metrics_unet.json missing; using hardcoded values")

    # FPS benchmark
    with open(RESULTS_DIR / "fps_benchmark.json") as f:
        fps = json.load(f)
    out["fps"] = {
        "classical": fps["results"][0]["avg_fps"],
        "ema":       fps["results"][1]["avg_fps"],
        "unet":      fps["results"][2]["avg_fps"],
        "device":    fps["device"],
        "n_frames":  fps["n_frames"],
    }

    # Temporal smoothing jitter
    jitter_path = RESULTS_DIR / "temporal_jitter_per_clip.csv"
    raw_c_vals = []
    sm_c_vals = []
    with open(jitter_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_c_vals.append(float(row["raw_c_jitter"]))
            sm_c_vals.append(float(row["sm_c_jitter"]))
    out["temporal"] = {
        "avg_raw_jitter":   float(np.mean(raw_c_vals)),
        "avg_sm_jitter":    float(np.mean(sm_c_vals)),
        "per_pair_raw":     raw_c_vals,
        "per_pair_smooth":  sm_c_vals,
        "reduction_factor": float(np.mean(raw_c_vals) / max(np.mean(sm_c_vals), 1e-9)),
        "n_pairs":          len(raw_c_vals),
    }

    return out


def draw_panel_a_table(ax, metrics: dict) -> None:
    """Panel A: TuSimple metrics table."""
    ax.axis("off")
    ax.set_title("(A) TuSimple Benchmark — Detection Quality",
                 fontsize=13, fontweight="bold", loc="left", pad=12)

    rows = [
        ["Method", "Accuracy", "FPR", "FNR"],
        [
            "Classical CV",
            f"{100 * metrics['classical']['accuracy']:.2f}%",
            f"{100 * metrics['classical']['fpr']:.2f}%",
            f"{100 * metrics['classical']['fnr']:.2f}%",
        ],
        [
            "U-Net",
            f"{100 * metrics['unet']['accuracy']:.2f}%",
            f"{100 * metrics['unet']['fpr']:.2f}%",
            f"{100 * metrics['unet']['fnr']:.2f}%",
        ],
    ]

    table = ax.table(
        cellText=rows[1:],
        colLabels=rows[0],
        cellLoc="center",
        loc="center",
        colColours=["#cccccc"] * 4,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.0)

    # Highlight better values per column
    classical_idx, unet_idx = 1, 2
    # Accuracy: higher is better
    if metrics["unet"]["accuracy"] >= metrics["classical"]["accuracy"]:
        table[(unet_idx, 1)].set_facecolor("#c8e6c9")
    else:
        table[(classical_idx, 1)].set_facecolor("#c8e6c9")
    # FPR: lower is better
    if metrics["unet"]["fpr"] < metrics["classical"]["fpr"]:
        table[(unet_idx, 2)].set_facecolor("#c8e6c9")
    else:
        table[(classical_idx, 2)].set_facecolor("#c8e6c9")
    # FNR: lower is better
    if metrics["unet"]["fnr"] < metrics["classical"]["fnr"]:
        table[(unet_idx, 3)].set_facecolor("#c8e6c9")
    else:
        table[(classical_idx, 3)].set_facecolor("#c8e6c9")

    ax.text(0.5, 0.05,
            f"Evaluated on {metrics['classical']['n_frames']} test frames.  "
            "Green = better.",
            transform=ax.transAxes, ha="center", fontsize=9, color="gray")


def draw_panel_b_fps(ax, metrics: dict) -> None:
    """Panel B: FPS bar chart for all three methods."""
    fps = metrics["fps"]
    methods = ["Classical CV", "Classical + EMA", "U-Net"]
    values = [fps["classical"], fps["ema"], fps["unet"]]
    colors = ["#4a90d9", "#2da77e", "#d97a4a"]

    bars = ax.bar(methods, values, color=colors,
                  edgecolor="black", linewidth=0.5)

    for bar, fps_val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                f"{fps_val:.1f} fps",
                ha="center", va="bottom", fontweight="bold", fontsize=10)

    # 30 fps reference line (real-time threshold)
    ax.axhline(30, linestyle="--", color="red", alpha=0.5, linewidth=1)
    ax.text(2.4, 31, "30 fps (real-time)", color="red",
            fontsize=8, ha="right", va="bottom")

    ax.set_ylabel("Average FPS", fontsize=11)
    ax.set_title(f"(B) Throughput on {metrics['fps']['device'].upper()} "
                 f"({metrics['fps']['n_frames']} frames)",
                 fontsize=13, fontweight="bold", loc="left", pad=12)
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=9)


def draw_panel_c_temporal(ax, metrics: dict) -> None:
    """Panel C: temporal smoothing jitter reduction (paired bar chart)."""
    temp = metrics["temporal"]
    n_pairs = temp["n_pairs"]
    indices = np.arange(n_pairs)
    width = 0.35

    raw_vals = temp["per_pair_raw"]
    sm_vals = temp["per_pair_smooth"]

    ax.bar(indices - width / 2, raw_vals, width,
           label="Raw (single-frame)", color="#d97a4a",
           edgecolor="black", linewidth=0.3)
    ax.bar(indices + width / 2, sm_vals, width,
           label="With EMA (ours)", color="#2da77e",
           edgecolor="black", linewidth=0.3)

    ax.set_xlabel("clip × lane pair (left/right)", fontsize=10)
    ax.set_ylabel("Jitter (std of frame-to-frame Δc, pixels)", fontsize=10)
    ax.set_title(f"(C) Temporal Smoothing reduces jitter "
                 f"({temp['reduction_factor']:.1f}× average reduction)",
                 fontsize=13, fontweight="bold", loc="left", pad=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticks([])  # too many to label individually

    # Annotate with averages
    avg_raw = temp["avg_raw_jitter"]
    avg_sm = temp["avg_sm_jitter"]
    ax.text(0.02, 0.95,
            f"avg raw jitter:    {avg_raw:.2f} px\n"
            f"avg smoothed jitter: {avg_sm:.2f} px\n"
            f"reduction:         {temp['reduction_factor']:.1f}×",
            transform=ax.transAxes, fontsize=9, family="monospace",
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray"))


def main() -> None:
    metrics = load_metrics()

    print("Loaded metrics:")
    print(f"  Classical: acc={100*metrics['classical']['accuracy']:.2f}%, "
          f"fpr={100*metrics['classical']['fpr']:.2f}%, "
          f"fnr={100*metrics['classical']['fnr']:.2f}%")
    print(f"  U-Net:     acc={100*metrics['unet']['accuracy']:.2f}%, "
          f"fpr={100*metrics['unet']['fpr']:.2f}%, "
          f"fnr={100*metrics['unet']['fnr']:.2f}%")
    print(f"  FPS:       classical={metrics['fps']['classical']:.1f}, "
          f"ema={metrics['fps']['ema']:.1f}, "
          f"unet={metrics['fps']['unet']:.1f}")
    print(f"  Temporal:  avg raw={metrics['temporal']['avg_raw_jitter']:.2f}, "
          f"avg smoothed={metrics['temporal']['avg_sm_jitter']:.2f}, "
          f"reduction={metrics['temporal']['reduction_factor']:.1f}x")
    print()

    # === Build the figure ===
    fig = plt.figure(figsize=(16, 10))

    # Title row
    fig.suptitle("Lane Detection Method Comparison on TuSimple Benchmark",
                 fontsize=16, fontweight="bold", y=0.98)

    # 2-row, 2-col grid:
    # Top row: panel A (full width)
    # Bottom row: panel B (left half), panel C (right half)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.5, 1.3],
                          hspace=0.35, wspace=0.25,
                          top=0.93, bottom=0.06, left=0.05, right=0.97)

    ax_a = fig.add_subplot(gs[0, :])       # full top row
    ax_b = fig.add_subplot(gs[1, 0])       # bottom-left
    ax_c = fig.add_subplot(gs[1, 1])       # bottom-right

    draw_panel_a_table(ax_a, metrics)
    draw_panel_b_fps(ax_b, metrics)
    draw_panel_c_temporal(ax_c, metrics)

    # Save as both PNG (for slides) and PDF (for LaTeX report)
    png_path = RESULTS_DIR / "summary_comparison.png"
    pdf_path = RESULTS_DIR / "summary_comparison.pdf"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"\nSaved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
