"""
TrendPulse — Task 3: Analyse with NumPy & Pandas
==================================================
Loads the cleaned CSV from Task 2 and produces a structured analysis:
  • Per-category descriptive statistics (score & comments)
  • Top 5 stories overall by score
  • Top author per category
  • Engagement ratio (comments / score) per category
  • Saves a summary CSV for Task 4

Pipeline: Task 1 (Fetch) → Task 2 (Clean CSV) → Task 3 (NumPy/Pandas) → Task 4 (Visualise)
"""

import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime


# ── Configuration ─────────────────────────────────────────────────────────────

INPUT_DIR  = "data"
OUTPUT_DIR = "data"


# ── Step 1: Load the cleaned CSV ──────────────────────────────────────────────

def find_latest_csv():
    """
    Glob for all trends_clean_*.csv files and return the most recent one,
    so the script always picks up today's Task 2 output automatically.
    """
    pattern = os.path.join(INPUT_DIR, "trends_clean_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No trends_clean_*.csv found in '{INPUT_DIR}/'. "
            "Run task2_data_cleaning.py first."
        )
    return sorted(files)[-1]


def load_csv(filepath):
    """
    Read the CSV into a DataFrame.
    Enforce correct dtypes so pandas doesn't guess wrong:
      - score / num_comments → integer
      - collected_at         → datetime
    """
    df = pd.read_csv(
        filepath,
        dtype={
            "post_id":      str,       # treat as identifier, not a number to sum
            "title":        str,
            "category":     str,
            "score":        int,
            "num_comments": int,
            "author":       str,
        },
        parse_dates=["collected_at"],  # pandas parses the timestamp string
    )
    print(f"Loaded {len(df)} rows from '{filepath}'.\n")
    return df


# ── Step 2: Basic validation ───────────────────────────────────────────────────

def validate(df):
    """
    Quick sanity checks using pandas / NumPy.
    Prints warnings but does not abort — Task 2 should have already cleaned well.
    """
    print("── Validation ──────────────────────────────────────────────────────")

    # Check for any remaining nulls
    nulls = df.isnull().sum()
    null_cols = nulls[nulls > 0]
    if null_cols.empty:
        print("  ✓ No null values found.")
    else:
        print(f"  ⚠ Null values detected:\n{null_cols}")

    # Check numeric ranges with NumPy
    neg_scores    = np.sum(df["score"].values < 0)
    neg_comments  = np.sum(df["num_comments"].values < 0)
    if neg_scores:
        print(f"  ⚠ {neg_scores} rows have negative scores.")
    if neg_comments:
        print(f"  ⚠ {neg_comments} rows have negative comment counts.")

    if not neg_scores and not neg_comments:
        print("  ✓ All numeric values are non-negative.")

    # Check category set
    cats = set(df["category"].unique())
    expected = {"technology", "worldnews", "sports", "science", "entertainment"}
    unexpected = cats - expected - {"unknown"}
    if unexpected:
        print(f"  ⚠ Unexpected categories found: {unexpected}")
    else:
        print(f"  ✓ Categories present: {sorted(cats)}")

    print()


# ── Step 3: Descriptive statistics ────────────────────────────────────────────

def category_stats(df):
    """
    Use pandas groupby + NumPy aggregations to build a stats table per category.
    Columns: count, mean_score, median_score, std_score,
             mean_comments, median_comments, max_score, total_comments
    """
    print("── Per-Category Statistics ─────────────────────────────────────────")

    grouped = df.groupby("category")

    stats = grouped.agg(
        count          = ("post_id",      "count"),
        mean_score     = ("score",        np.mean),       # NumPy function
        median_score   = ("score",        np.median),     # NumPy function
        std_score      = ("score",        np.std),        # NumPy function
        max_score      = ("score",        np.max),
        mean_comments  = ("num_comments", np.mean),
        median_comments= ("num_comments", np.median),
        total_comments = ("num_comments", np.sum),
    ).round(2)

    # Add engagement ratio: average comments per score point
    # Guard against division by zero with np.where
    stats["engagement_ratio"] = np.where(
        stats["mean_score"] > 0,
        (stats["mean_comments"] / stats["mean_score"]).round(4),
        0
    )

    print(stats.to_string())
    print()
    return stats


# ── Step 4: Top stories ────────────────────────────────────────────────────────

def top_stories_overall(df, n=5):
    """Return the top-n stories by score across all categories."""
    print(f"── Top {n} Stories by Score (Overall) ──────────────────────────────")
    top = (
        df[["title", "category", "score", "num_comments", "author"]]
        .sort_values("score", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    top.index += 1   # rank from 1
    print(top.to_string())
    print()
    return top


def top_story_per_category(df):
    """Return the single highest-scoring story for each category."""
    print("── Top Story per Category ──────────────────────────────────────────")
    idx  = df.groupby("category")["score"].idxmax()
    top  = df.loc[idx, ["category", "title", "score", "author"]].reset_index(drop=True)
    top  = top.sort_values("category")
    print(top.to_string(index=False))
    print()
    return top


def top_author_per_category(df):
    """
    Find the most prolific author in each category (by story count).
    Ties are broken by total score — the higher scorer wins.
    """
    print("── Most Prolific Author per Category ───────────────────────────────")
    author_stats = (
        df.groupby(["category", "author"])
        .agg(stories=("post_id", "count"), total_score=("score", "sum"))
        .reset_index()
        .sort_values(["category", "stories", "total_score"], ascending=[True, False, False])
    )
    # Pick the first (best) author per category
    top_authors = author_stats.groupby("category").first().reset_index()
    print(top_authors.to_string(index=False))
    print()
    return top_authors


# ── Step 5: NumPy-powered extra insights ──────────────────────────────────────

def numpy_insights(df):
    """
    Demonstrate direct NumPy array operations on the data:
      • Overall score percentiles
      • Z-scores to flag viral stories (|z| > 2)
      • Correlation between score and comment count
    """
    print("── NumPy Insights ──────────────────────────────────────────────────")

    scores = df["score"].values.astype(float)

    # Percentiles
    p25, p50, p75, p90 = np.percentile(scores, [25, 50, 75, 90])
    print(f"  Score percentiles → P25: {p25:.0f}  P50: {p50:.0f}  "
          f"P75: {p75:.0f}  P90: {p90:.0f}")

    # Z-scores — identify statistically viral stories
    mean  = np.mean(scores)
    std   = np.std(scores)
    z     = (scores - mean) / std if std > 0 else np.zeros_like(scores)
    viral = df[np.abs(z) > 2][["title", "category", "score"]]
    print(f"\n  Stories with |z-score| > 2 (statistically viral) — {len(viral)} found:")
    if not viral.empty:
        print(viral.to_string(index=False))
    else:
        print("  (none — distribution is fairly flat)")

    # Pearson correlation: score vs num_comments
    comments = df["num_comments"].values.astype(float)
    correlation = np.corrcoef(scores, comments)[0, 1]
    print(f"\n  Pearson correlation (score ↔ comments): {correlation:.4f}")
    if correlation > 0.5:
        print("  → Strong positive relationship: high-score posts attract more comments.")
    elif correlation > 0.2:
        print("  → Moderate positive relationship.")
    else:
        print("  → Weak relationship: score and comments are fairly independent.")

    print()
    return viral


# ── Step 6: Save summary for Task 4 ───────────────────────────────────────────

def save_summary(stats, df):
    """
    Persist two artefacts for Task 4:
      1. data/category_summary_YYYYMMDD.csv  — per-category stats table
      2. data/trends_analysed_YYYYMMDD.csv   — full DataFrame with z-score column
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")

    # Summary stats (one row per category)
    summary_path = os.path.join(OUTPUT_DIR, f"category_summary_{date_str}.csv")
    stats.to_csv(summary_path)
    print(f"Category summary saved → '{summary_path}'")

    # Full dataset enriched with a z-score column (useful for Task 4 colouring)
    scores       = df["score"].values.astype(float)
    mean, std    = np.mean(scores), np.std(scores)
    df = df.copy()
    df["score_zscore"] = np.round((scores - mean) / std, 4) if std > 0 else 0.0

    analysed_path = os.path.join(OUTPUT_DIR, f"trends_analysed_{date_str}.csv")
    df.to_csv(analysed_path, index=False)
    print(f"Enriched dataset saved  → '{analysed_path}'")

    return summary_path, analysed_path


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== TrendPulse — Task 3: Analysis ===\n")

    # 1. Load data
    csv_path = find_latest_csv()
    df = load_csv(csv_path)

    # 2. Validate
    validate(df)

    # 3. Per-category descriptive stats (uses NumPy aggregations via pandas)
    stats = category_stats(df)

    # 4. Story-level rankings
    top_stories_overall(df, n=5)
    top_story_per_category(df)
    top_author_per_category(df)

    # 5. Raw NumPy array operations
    numpy_insights(df)

    # 6. Persist outputs for Task 4
    save_summary(stats, df)

    print("\nTask 3 complete. Pass the CSVs above to task4_visualisation.py.")