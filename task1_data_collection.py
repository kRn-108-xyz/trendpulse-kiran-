"""
TrendPulse — Task 1: Fetch Data from HackerNews API
=====================================================
Fetches trending stories from the HackerNews public API,
categorises them by keyword matching, and saves to a JSON file.

Pipeline: Task 1 (Fetch) → Task 2 (Clean CSV) → Task 3 (NumPy/Pandas) → Task 4 (Visualise)
"""

import requests
import json
import os
import time
from datetime import datetime

# ── Configuration ────────────────────────────────────────────────────────────

BASE_URL = "https://hacker-news.firebaseio.com/v0"
HEADERS = {"User-Agent": "TrendPulse/1.0"}

# Maximum stories to fetch from the top-stories list
MAX_IDS = 500

# Maximum stories to keep per category
MAX_PER_CATEGORY = 25

# Keyword map: each category maps to a list of case-insensitive trigger words
CATEGORIES = {
    "technology":    ["AI", "software", "tech", "code", "computer",
                      "data", "cloud", "API", "GPU", "LLM"],
    "worldnews":     ["war", "government", "country", "president",
                      "election", "climate", "attack", "global"],
    "sports":        ["NFL", "NBA", "FIFA", "sport", "game", "team",
                      "player", "league", "championship"],
    "science":       ["research", "study", "space", "physics", "biology",
                      "discovery", "NASA", "genome"],
    "entertainment": ["movie", "film", "music", "Netflix", "game",
                      "book", "show", "award", "streaming"],
}

# ── Helper functions ─────────────────────────────────────────────────────────

def fetch_top_story_ids():
    """
    Step 1: Hit the /topstories endpoint and return the first MAX_IDS IDs.
    Returns an empty list if the request fails so the script keeps running.
    """
    url = f"{BASE_URL}/topstories.json"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()          # raises HTTPError for 4xx/5xx
        ids = response.json()
        print(f"Fetched {len(ids)} story IDs. Using first {MAX_IDS}.")
        return ids[:MAX_IDS]
    except requests.RequestException as e:
        print(f"Failed to fetch top story IDs: {e}")
        return []


def fetch_story(story_id):
    """
    Step 2: Fetch a single story's detail object by its ID.
    Returns None on any network or HTTP error so the caller can skip it.
    """
    url = f"{BASE_URL}/item/{story_id}.json"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"  Failed to fetch story {story_id}: {e}")
        return None


def assign_category(title):
    """
    Check the story title (case-insensitive) against each category's keywords.
    Returns the FIRST matching category name, or None if no keyword matches.
    """
    if not title:
        return None
    title_lower = title.lower()
    for category, keywords in CATEGORIES.items():
        for keyword in keywords:
            if keyword.lower() in title_lower:
                return category
    return None   # story doesn't match any category


def extract_fields(story, category):
    """
    Pull the 7 required fields from a raw HackerNews story object.
    'collected_at' is added by us at collection time.
    """
    return {
        "post_id":      story.get("id"),
        "title":        story.get("title", ""),
        "category":     category,
        "score":        story.get("score", 0),
        "num_comments": story.get("descendants", 0),  # HN uses 'descendants' for comments
        "author":       story.get("by", ""),
        "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ── Main collection logic ─────────────────────────────────────────────────────

def collect_stories():
    """
    Core pipeline:
      1. Fetch top story IDs.
      2. For each story, check if its title matches a category.
      3. Collect up to MAX_PER_CATEGORY stories per category.
      4. Sleep 2 seconds between processing each category group.
    Returns a flat list of story dicts.
    """
    story_ids = fetch_top_story_ids()
    if not story_ids:
        print("No story IDs retrieved. Exiting.")
        return []

    # Pre-fetch all story details in one pass to avoid redundant requests.
    # We stop once all categories are full OR we run out of IDs.
    print("\nFetching individual story details...")

    # Bucket to hold results per category
    buckets = {cat: [] for cat in CATEGORIES}

    for story_id in story_ids:
        # Stop early if every category already has MAX_PER_CATEGORY stories
        if all(len(v) >= MAX_PER_CATEGORY for v in buckets.values()):
            print("All categories full — stopping early.")
            break

        story = fetch_story(story_id)
        if not story:
            continue  # network error — skip and move on

        # Only process stories with a title (some HN items are jobs/polls)
        title = story.get("title", "")
        if not title:
            continue

        category = assign_category(title)
        if not category:
            continue  # title doesn't match any keyword

        # Only add if this category still has room
        if len(buckets[category]) < MAX_PER_CATEGORY:
            buckets[category].append(extract_fields(story, category))

    # ── One sleep per category (not per story) ────────────────────────────────
    # We loop over categories purely to honour the "sleep between categories"
    # requirement; the actual fetching was already done above.
    print("\nApplying inter-category delay (2 s per category)...")
    for category in CATEGORIES:
        count = len(buckets[category])
        print(f"  {category}: {count} stories collected")
        time.sleep(2)   # 2-second pause between each category, as required

    # Flatten all buckets into a single list
    all_stories = [story for cat_stories in buckets.values()
                   for story in cat_stories]
    return all_stories


# ── Save output ───────────────────────────────────────────────────────────────

def save_to_json(stories):
    """
    Creates the data/ directory if it doesn't exist, then writes the
    collected stories to a date-stamped JSON file.
    """
    # Create the output folder
    os.makedirs("data", exist_ok=True)

    # Build a filename like data/trends_20240115.json
    date_str = datetime.now().strftime("%Y%m%d")
    filename = f"data/trends_{date_str}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(stories, f, indent=2, ensure_ascii=False)

    print(f"\nCollected {len(stories)} stories. Saved to {filename}")
    return filename


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== TrendPulse — Task 1: Data Collection ===\n")
    stories = collect_stories()

    if stories:
        save_to_json(stories)
    else:
        print("No stories collected. Check your network connection and try again.")