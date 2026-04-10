"""
TrendPulse — Task 2: Clean Data & Save as CSV
===============================================
Loads the JSON file produced by Task 1, cleans and validates every field,
then writes a tidy CSV ready for Task 3 (NumPy / Pandas analysis).

Pipeline: Task 1 (Fetch) → Task 2 (Clean CSV) → Task 3 (NumPy/Pandas) → Task 4 (Visualise)
"""

import json
import csv
import os
import glob
from datetime import datetime


# ── Configuration ─────────────────────────────────────────────────────────────

# Folder that Task 1 wrote into
INPUT_DIR = "data"

# Output CSV will also live in data/
OUTPUT_DIR = "data"

# Valid category values — anything else gets relabelled "unknown"
VALID_CATEGORIES = {"technology", "worldnews", "sports", "science", "entertainment"}

# CSV column order (matches Task 1's 7 fields exactly)
CSV_FIELDS = ["post_id", "title", "category", "score", "num_comments", "author", "collected_at"]


# ── Step 1: Load JSON ─────────────────────────────────────────────────────────

def find_latest_json():
    """
    Glob for all trends_*.json files in data/ and return the most recent one.
    This way the script always picks up Task 1's output even if the date differs.
    """
    pattern = os.path.join(INPUT_DIR, "trends_*.json")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No trends_*.json file found in '{INPUT_DIR}/'. "
            "Run task1_data_collection.py first."
        )
    # Sort by filename (date-stamped) and take the latest
    latest = sorted(files)[-1]
    return latest


def load_json(filepath):
    """Read the JSON file and return the list of story dicts."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} raw stories from '{filepath}'.")
    return data


# ── Step 2: Clean & validate each record ──────────────────────────────────────

def clean_title(title):
    """
    Strip leading/trailing whitespace.
    Replace any internal runs of whitespace with a single space.
    Return a fallback string if the result is empty.
    """
    if not isinstance(title, str):
        return "untitled"
    cleaned = " ".join(title.split())   # collapses all whitespace
    return cleaned if cleaned else "untitled"


def clean_integer(value, field_name, default=0):
    """
    Coerce value to a non-negative integer.
    Prints a warning and returns `default` if coercion fails or value < 0.
    """
    try:
        result = int(value)
        if result < 0:
            print(f"  Warning: negative {field_name} ({result}), resetting to {default}.")
            return default
        return result
    except (TypeError, ValueError):
        print(f"  Warning: invalid {field_name} '{value}', resetting to {default}.")
        return default


def clean_category(category):
    """Return category as-is if valid, otherwise 'unknown'."""
    if isinstance(category, str) and category.strip().lower() in VALID_CATEGORIES:
        return category.strip().lower()
    return "unknown"


def clean_author(author):
    """Strip whitespace; replace missing/empty with 'anonymous'."""
    if not isinstance(author, str) or not author.strip():
        return "anonymous"
    return author.strip()


def clean_collected_at(value):
    """
    Validate the timestamp string.  If it already parses correctly, keep it.
    Otherwise stamp it with the current time so no row is left blank.
    """
    fmt = "%Y-%m-%d %H:%M:%S"
    try:
        datetime.strptime(value, fmt)   # just validate — re-raise on failure
        return value
    except (TypeError, ValueError):
        now = datetime.now().strftime(fmt)
        print(f"  Warning: bad collected_at '{value}', replacing with '{now}'.")
        return now


def clean_post_id(value):
    """Ensure post_id is a positive integer string; return '' if not."""
    try:
        pid = int(value)
        if pid > 0:
            return pid
    except (TypeError, ValueError):
        pass
    print(f"  Warning: invalid post_id '{value}' — row will be dropped.")
    return None


def clean_story(raw):
    """
    Apply all cleaning rules to a single story dict.
    Returns a cleaned dict, or None if the record is unsalvageable
    (e.g. missing post_id or title).
    """
    post_id = clean_post_id(raw.get("post_id"))
    if post_id is None:
        return None     # can't keep a row with no valid ID

    title = clean_title(raw.get("title"))
    if title == "untitled":
        # A story with no title isn't useful — drop it
        print(f"  Dropping story {post_id}: empty title.")
        return None

    return {
        "post_id":      post_id,
        "title":        title,
        "category":     clean_category(raw.get("category")),
        "score":        clean_integer(raw.get("score"),        "score"),
        "num_comments": clean_integer(raw.get("num_comments"), "num_comments"),
        "author":       clean_author(raw.get("author")),
        "collected_at": clean_collected_at(raw.get("collected_at")),
    }


def remove_duplicates(stories):
    """
    Deduplicate by post_id — keep only the first occurrence.
    This guards against the same story matching multiple categories in edge cases.
    """
    seen = set()
    unique = []
    for story in stories:
        if story["post_id"] not in seen:
            seen.add(story["post_id"])
            unique.append(story)
        else:
            print(f"  Duplicate post_id {story['post_id']} removed.")
    return unique


def clean_all(raw_stories):
    """
    Run every record through clean_story(), drop Nones, then deduplicate.
    Prints a summary of how many records were dropped.
    """
    print("\nCleaning records...")
    cleaned = []
    dropped = 0

    for raw in raw_stories:
        result = clean_story(raw)
        if result:
            cleaned.append(result)
        else:
            dropped += 1

    print(f"  Dropped {dropped} invalid records.")
    before_dedup = len(cleaned)
    cleaned = remove_duplicates(cleaned)
    dupes_removed = before_dedup - len(cleaned)
    if dupes_removed:
        print(f"  Removed {dupes_removed} duplicate post IDs.")

    print(f"  {len(cleaned)} clean records remain.")
    return cleaned


# ── Step 3: Save to CSV ───────────────────────────────────────────────────────

def save_to_csv(stories):
    """
    Write the cleaned stories to data/trends_clean_YYYYMMDD.csv.
    Uses csv.DictWriter so column order is always consistent.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")
    filename = os.path.join(OUTPUT_DIR, f"trends_clean_{date_str}.csv")

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(stories)

    print(f"\nCleaned {len(stories)} stories saved to '{filename}'.")
    return filename


# ── Step 4: Summary report ────────────────────────────────────────────────────

def print_summary(stories):
    """Print a quick per-category breakdown so results are easy to verify."""
    counts = {}
    for story in stories:
        cat = story["category"]
        counts[cat] = counts.get(cat, 0) + 1

    print("\n── Category breakdown ──────────────────────")
    for cat, n in sorted(counts.items()):
        print(f"  {cat:<15} {n:>3} stories")
    print(f"  {'TOTAL':<15} {len(stories):>3} stories")
    print("────────────────────────────────────────────")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== TrendPulse — Task 2: Data Cleaning ===\n")

    # 1. Find and load the Task 1 JSON output
    json_path = find_latest_json()
    raw_stories = load_json(json_path)

    # 2. Clean every record
    clean_stories = clean_all(raw_stories)

    # 3. Save to CSV
    csv_path = save_to_csv(clean_stories)

    # 4. Show a summary
    print_summary(clean_stories)

    print("\nTask 2 complete. Pass the CSV above to task3_analysis.py.")
    
    # hello world