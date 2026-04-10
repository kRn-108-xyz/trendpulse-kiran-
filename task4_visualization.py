"""
TrendPulse — Task 4: Visualise the Data
=========================================
Loads the two CSVs produced by Task 3 and generates a 6-panel dashboard
saved as a single PNG: data/trendpulse_dashboard_YYYYMMDD.png

Charts produced:
  1. Bar chart      — Story count per category
  2. Bar chart      — Mean score per category
  3. Horizontal bar — Top 10 stories by score
  4. Scatter plot   — Score vs Comments (coloured by category)
  5. Bar chart      — Engagement ratio per category
  6. Box plot       — Score distribution per category

Pipeline: Task 1 (Fetch) → Task 2 (Clean CSV) → Task 3 (NumPy/Pandas) → Task 4 (Visualise)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                   # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob
import os
from datetime import datetime


# ── Configuration ─────────────────────────────────────────────────────────────

INPUT_DIR  = "data"
OUTPUT_DIR = "data"

# One consistent colour per category across every chart
CATEGORY_COLOURS = {
    "technology":    "#4C72B0",
    "worldnews":     "#DD8452",
    "sports":        "#55A868",
    "science":       "#C44E52",
    "entertainment": "#8172B2",
    "unknown":       "#999999",
}

# Overall figure style
plt.rcParams.update({
    "figure.facecolor": "#F8F9FA",
    "axes.facecolor":   "#FFFFFF",
    "axes.grid":        True,
    "grid.alpha":       0.4,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "DejaVu Sans",
})


# ── Step 1: Load data ─────────────────────────────────────────────────────────

def find_latest(pattern):
    """Return the most recently dated file matching glob pattern."""
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No file matching '{pattern}'. Run earlier tasks first.")
    return sorted(files)[-1]


def load_data():
    """Load the category summary and the full enriched dataset from Task 3."""
    summary_path  = find_latest(os.path.join(INPUT_DIR, "category_summary_*.csv"))
    analysed_path = find_latest(os.path.join(INPUT_DIR, "trends_analysed_*.csv"))

    summary  = pd.read_csv(summary_path,  index_col=0)
    analysed = pd.read_csv(analysed_path, parse_dates=["collected_at"])

    print(f"Summary  loaded from '{summary_path}'  ({len(summary)} categories)")
    print(f"Analysed loaded from '{analysed_path}' ({len(analysed)} stories)\n")
    return summary, analysed


# ── Step 2: Individual chart functions ────────────────────────────────────────

def chart_story_count(ax, summary):
    """Bar chart: number of stories collected per category."""
    cats   = summary.index.tolist()
    counts = summary["count"].values
    colours = [CATEGORY_COLOURS.get(c, "#999") for c in cats]

    bars = ax.bar(cats, counts, color=colours, edgecolor="white", linewidth=0.8)

    # Add value labels on top of each bar
    for bar, val in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(int(val)),
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    ax.set_title("Stories Collected per Category", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xticklabels(cats, rotation=20, ha="right")
    ax.set_ylim(0, counts.max() * 1.2)


def chart_mean_score(ax, summary):
    """Bar chart: mean score per category with error bars (std)."""
    cats       = summary.index.tolist()
    means      = summary["mean_score"].values
    stds       = summary["std_score"].values
    colours    = [CATEGORY_COLOURS.get(c, "#999") for c in cats]

    bars = ax.bar(
        cats, means, yerr=stds,
        color=colours, edgecolor="white", linewidth=0.8,
        capsize=4, error_kw={"elinewidth": 1.2, "ecolor": "#555"}
    )

    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + stds[list(means).index(val)] + 1,
            f"{val:.0f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    ax.set_title("Mean Score per Category (± std)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Score")
    ax.set_xticklabels(cats, rotation=20, ha="right")


def chart_top10_stories(ax, df):
    """Horizontal bar chart: top 10 stories by score."""
    top10 = df.nlargest(10, "score")[["title", "category", "score"]].reset_index(drop=True)

    # Shorten titles to 55 chars so they fit the axis
    labels  = [t[:55] + "…" if len(t) > 55 else t for t in top10["title"]]
    colours = [CATEGORY_COLOURS.get(c, "#999") for c in top10["category"]]
    y_pos   = np.arange(len(top10))

    ax.barh(y_pos, top10["score"].values, color=colours, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()   # highest score on top
    ax.set_xlabel("Score")
    ax.set_title("Top 10 Stories by Score", fontsize=12, fontweight="bold")

    # Legend patches
    seen = {}
    for cat, col in zip(top10["category"], colours):
        seen[cat] = col
    patches = [mpatches.Patch(color=v, label=k) for k, v in seen.items()]
    ax.legend(handles=patches, fontsize=7, loc="lower right")


def chart_scatter(ax, df):
    """Scatter: score vs num_comments, coloured by category."""
    for cat, group in df.groupby("category"):
        colour = CATEGORY_COLOURS.get(cat, "#999")
        ax.scatter(
            group["score"], group["num_comments"],
            color=colour, label=cat,
            alpha=0.7, edgecolors="white", linewidths=0.4, s=50
        )

    # Overlay linear trend line (NumPy polyfit)
    scores   = df["score"].values.astype(float)
    comments = df["num_comments"].values.astype(float)
    if scores.std() > 0:
        m, b   = np.polyfit(scores, comments, 1)
        x_line = np.linspace(scores.min(), scores.max(), 200)
        ax.plot(x_line, m * x_line + b, color="#333", linewidth=1.2,
                linestyle="--", label="trend")

    # Pearson correlation in subtitle
    corr = np.corrcoef(scores, comments)[0, 1]
    ax.set_title(
        f"Score vs Comments  (r = {corr:.3f})",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Score")
    ax.set_ylabel("Comments")
    ax.legend(fontsize=7, markerscale=0.9)


def chart_engagement(ax, summary):
    """Bar chart: engagement ratio (mean_comments / mean_score) per category."""
    cats   = summary.index.tolist()
    ratios = summary["engagement_ratio"].values
    colours = [CATEGORY_COLOURS.get(c, "#999") for c in cats]

    bars = ax.bar(cats, ratios, color=colours, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, ratios):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    ax.set_title("Engagement Ratio (Comments / Score)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Ratio")
    ax.set_xticklabels(cats, rotation=20, ha="right")


def chart_boxplot(ax, df):
    """Box plot: score distribution per category."""
    cats = sorted(df["category"].unique())
    data = [df.loc[df["category"] == c, "score"].values for c in cats]

    bp = ax.boxplot(
        data,
        patch_artist=True,
        notch=False,
        medianprops={"color": "#222", "linewidth": 1.8},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.5},
    )
    # Apply category colours to boxes
    for patch, cat in zip(bp["boxes"], cats):
        patch.set_facecolor(CATEGORY_COLOURS.get(cat, "#999"))
        patch.set_alpha(0.8)

    ax.set_xticks(range(1, len(cats) + 1))
    ax.set_xticklabels(cats, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Score Distribution per Category", fontsize=12, fontweight="bold")


# ── Step 3: Compose the dashboard ─────────────────────────────────────────────

def build_dashboard(summary, df):
    """
    Arrange all 6 charts in a 3×2 grid and save as a high-res PNG.
    """
    fig, axes = plt.subplots(
        nrows=3, ncols=2,
        figsize=(16, 18),
        facecolor="#F0F2F5"
    )
    fig.suptitle(
        "TrendPulse — HackerNews Trend Dashboard",
        fontsize=18, fontweight="bold", y=0.98, color="#1A1A2E"
    )

    # Map each axis to its chart function
    chart_story_count   (axes[0, 0], summary)
    chart_mean_score    (axes[0, 1], summary)
    chart_top10_stories (axes[1, 0], df)
    chart_scatter       (axes[1, 1], df)
    chart_engagement    (axes[2, 0], summary)
    chart_boxplot       (axes[2, 1], df)

    # Tight layout with room for suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    date_str  = datetime.now().strftime("%Y%m%d")
    out_path  = os.path.join(OUTPUT_DIR, f"trendpulse_dashboard_{date_str}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Dashboard saved → '{out_path}'")
    return out_path


# ── Step 4: Plain-text insight summary ────────────────────────────────────────

def print_insights(summary, df):
    """Print a human-readable highlight reel after the charts are saved."""
    print("\n── Key Insights ────────────────────────────────────────────────────")

    best_cat   = summary["mean_score"].idxmax()
    worst_cat  = summary["mean_score"].idxmin()
    engage_cat = summary["engagement_ratio"].idxmax()
    top_story  = df.nlargest(1, "score").iloc[0]

    print(f"  • Highest avg score  : {best_cat}  ({summary.loc[best_cat, 'mean_score']:.0f} pts)")
    print(f"  • Lowest  avg score  : {worst_cat}  ({summary.loc[worst_cat, 'mean_score']:.0f} pts)")
    print(f"  • Most engaging cat  : {engage_cat}  "
          f"(ratio {summary.loc[engage_cat, 'engagement_ratio']:.3f})")
    print(f"  • Top single story   : \"{top_story['title'][:60]}\"")
    print(f"    Score {top_story['score']}  |  "
          f"{top_story['num_comments']} comments  |  @{top_story['author']}")
    print("────────────────────────────────────────────────────────────────────")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== TrendPulse — Task 4: Visualisation ===\n")

    # 1. Load Task 3 outputs
    summary, df = load_data()

    # 2. Build and save the 6-panel dashboard
    build_dashboard(summary, df)

    # 3. Print a short insight summary to the console
    print_insights(summary, df)

    print("\nTask 4 complete. Pipeline finished!")
    print("Output: data/trendpulse_dashboard_YYYYMMDD.png")