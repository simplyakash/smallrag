import argparse
import csv
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


# This script builds a very small Amazon-style recommendation MVP.
# It has three jobs:
# 1. Download or create user-product interaction data.
# 2. Convert that data into item-to-item similarity scores.
# 3. Print product recommendations for a selected user.

# Local folder where the downloaded/sample dataset will be written.
DATA_DIR = Path("data/amazon_recommendation")

# CSV file containing user interactions, for example purchases or views.
EVENTS_FILE = DATA_DIR / "events.csv"

# CSV file containing product details such as product title and category.
PRODUCTS_FILE = DATA_DIR / "products.csv"

# Direct URL to a small Amazon Reviews 2023 JSONL file.
# We use the direct file URL because some Hugging Face dataset loaders are
# deprecated in newer versions of the `datasets` package.
DEFAULT_DATA_FILE_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
    "raw/review_categories/All_Beauty.jsonl"
)

# Tiny built-in interaction dataset used when you run:
# python -m src.amazon_recommendation_mvp --sample
#
# Each row means: this user interacted with this product.
# In this MVP, all sample events are purchases because purchases are a strong
# shopping recommendation signal.

SAMPLE_EVENTS = [
    {"user_id": "u1", "product_id": "p1", "event_type": "purchase", "rating": "5", "timestamp": "2026-01-01"},
    {"user_id": "u1", "product_id": "p2", "event_type": "purchase", "rating": "4", "timestamp": "2026-01-02"},
    {"user_id": "u2", "product_id": "p1", "event_type": "purchase", "rating": "5", "timestamp": "2026-01-03"},
    {"user_id": "u2", "product_id": "p3", "event_type": "purchase", "rating": "5", "timestamp": "2026-01-04"},
    {"user_id": "u3", "product_id": "p2", "event_type": "purchase", "rating": "4", "timestamp": "2026-01-05"},
    {"user_id": "u3", "product_id": "p4", "event_type": "purchase", "rating": "5", "timestamp": "2026-01-06"},
    {"user_id": "u4", "product_id": "p3", "event_type": "purchase", "rating": "4", "timestamp": "2026-01-07"},
    {"user_id": "u4", "product_id": "p5", "event_type": "purchase", "rating": "5", "timestamp": "2026-01-08"},
]

# Tiny built-in product catalog for the sample dataset above.
# The real downloaded dataset gives us product IDs and ratings, but product
# metadata can be limited, so this sample keeps titles easy to read.

SAMPLE_PRODUCTS = {
    "p1": {"product_id": "p1", "title": "Wireless Mouse", "category": "Electronics"},
    "p2": {"product_id": "p2", "title": "USB Keyboard", "category": "Electronics"},
    "p3": {"product_id": "p3", "title": "Running Shoes", "category": "Sports"},
    "p4": {"product_id": "p4", "title": "Laptop Stand", "category": "Office"},
    "p5": {"product_id": "p5", "title": "Water Bottle", "category": "Sports"},
}

# Create the local data directory if it does not already exist.
# This keeps all generated CSV files under `data/amazon_recommendation`.
# `parents=True` creates parent folders as needed, and `exist_ok=True` avoids
# errors when the folder already exists.

def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

# Write a list of dictionaries to a CSV file.
# `fieldnames` controls the output column order.
# This helper is reused for both `events.csv` and `products.csv`.

def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    # Make sure the target folder exists before opening the file.
    ensure_data_dir()

    # `newline=""` prevents extra blank lines on some platforms.
    # `utf-8` lets product titles and text fields be written safely.
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write the header row first, then the data rows.
        writer.writeheader()
        writer.writerows(rows)

# Convert one raw Amazon review row into two clean records:
# 1. An event record: user_id, product_id, event_type, rating, timestamp.
# 2. A product record: product_id, title, category.
#
# Different Amazon datasets use different field names, so this function checks
# several possible names for the same concept, such as `asin` or `product_id`.

def normalize_amazon_review(row: dict) -> tuple[dict, dict] | None:
    # Try all known user ID field names.
    user_id = row.get("customer_id") or row.get("reviewerID") or row.get("user_id")

    # Try all known product ID field names.
    # `asin` is Amazon's product identifier.

    product_id = row.get("product_id") or row.get("asin") or row.get("parent_asin")

    # If either key is missing, we cannot use the row for recommendations.
    if not user_id or not product_id:
        return None

    # Ratings and timestamps also vary by dataset version.
    rating = str(row.get("star_rating") or row.get("overall") or row.get("rating") or "")
    timestamp = str(row.get("review_date") or row.get("unixReviewTime") or row.get("timestamp") or "")

    # Raw review files usually do not include full catalog metadata, so use a
    # readable fallback title when product title is unavailable.
    title = row.get("product_title") or f"Product {product_id}"
    category = row.get("product_category") or row.get("main_category") or "unknown"

    # Treat a review as a purchase-like interaction.
    # This is reasonable for Amazon review data because users normally review
    # products they bought or used.
    event = {
        "user_id": str(user_id),
        "product_id": str(product_id),
        "event_type": "purchase",
        "rating": rating,
        "timestamp": timestamp,
    }

    # Keep a small product catalog so final recommendations can print names.
    product = {
        "product_id": str(product_id),
        "title": str(title),
        "category": str(category),
    }
    return event, product


# Download a small sample of Amazon review interactions from Hugging Face.
# The output is two local CSV files:
# - `events.csv` for recommendation signals.
# - `products.csv` for product display information.
#
# `limit` controls how many interactions we keep so the script runs quickly
# on a laptop or small development machine.

def download_amazon_reviews(
    limit: int,
    dataset: str,
    config: str,
    split: str,
    data_file_url: str,
) -> None:
    # Import here so the rest of the script can still run in sample mode even
    # if the optional Hugging Face dependency is not installed.
    
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Install recommendation dependencies first: "
            "python -m pip install -r requirements-recommendation.txt"
        ) from exc

    # Prefer the direct JSONL file URL. It avoids older Hugging Face dataset
    # loader scripts that may not work with the installed `datasets` version.
    if data_file_url:
        stream = load_dataset("json", data_files=data_file_url, split="train", streaming=True)
    else:
        # Fallback path for users who want to load another dataset/config.
        stream = load_dataset(dataset, config, split=split, streaming=True)

    # `events` becomes the recommender training input.
    events = []

    # `products` is a dictionary keyed by product_id so duplicate products are
    # stored only once.
    products = {}

    # Streaming reads the remote dataset row by row instead of downloading the
    # whole file before processing.
    for row in stream:
        normalized = normalize_amazon_review(row)
        if normalized is None:
            continue

        event, product = normalized
        events.append(event)
        products[product["product_id"]] = product

        # Stop once we have enough rows for the local prototype.
        if len(events) >= limit:
            break

    # If no rows could be parsed, the selected dataset is not compatible with
    # the expected Amazon review schema.
    if not events:
        raise RuntimeError("No usable interactions were downloaded from the selected dataset.")

    # Persist the downloaded sample as local CSV files so future runs can use
    # the data without downloading again.
    write_csv(EVENTS_FILE, events, ["user_id", "product_id", "event_type", "rating", "timestamp"])
    write_csv(PRODUCTS_FILE, list(products.values()), ["product_id", "title", "category"])

    print(f"Downloaded {len(events)} interactions to {EVENTS_FILE}")
    print(f"Saved {len(products)} products to {PRODUCTS_FILE}")

    # Show a few real user IDs so you can copy one into `--user-id`.
    sample_user_ids = sorted({event["user_id"] for event in events})[:5]
    print("Sample user IDs:", ", ".join(sample_user_ids))


# Write the tiny built-in dataset to disk.
# This is useful for testing the recommendation code without internet access
# or without installing the Hugging Face `datasets` package.
def write_sample_data() -> None:
    write_csv(EVENTS_FILE, SAMPLE_EVENTS, ["user_id", "product_id", "event_type", "rating", "timestamp"])
    write_csv(PRODUCTS_FILE, list(SAMPLE_PRODUCTS.values()), ["product_id", "title", "category"])
    print(f"Wrote sample interactions to {EVENTS_FILE}")
    print(f"Wrote sample products to {PRODUCTS_FILE}")


# Load a CSV file into a list of dictionaries.
# Each row becomes one dictionary where keys are column names.
def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


# Load product metadata from `products.csv`.
# Returning a dictionary by product_id makes lookup fast when printing final
# recommendations.
def load_products(path: Path) -> dict[str, dict]:
    # If the product catalog does not exist, recommendations can still print
    # product IDs without titles/categories.
    if not path.exists():
        return {}

    return {row["product_id"]: row for row in load_csv(path)}


# Convert one interaction into a numeric strength score.
# Stronger actions should matter more:
# - view is weak
# - add_to_cart is stronger
# - purchase is strongest
#
# Ratings above 3 add extra positive signal.
def interaction_weight(row: dict) -> float:
    # Assign a base score by event type.
    event_weight = {
        "view": 1.0,
        "add_to_cart": 2.0,
        "purchase": 3.0,
    }.get(row.get("event_type", ""), 1.0)

    # Convert rating text to a number. If rating is missing or invalid, use 0.
    try:
        rating = float(row.get("rating") or 0)
    except ValueError:
        rating = 0

    # A 4-star rating adds 1 point, a 5-star rating adds 2 points.
    # Ratings of 3 or below do not add positive signal.
    return event_weight + max(rating - 3.0, 0.0)


# Build item-to-item similarity from user interaction history.
#
# Main idea:
# If many users interact with both Product A and Product B, then A and B are
# probably related. This is a simple collaborative filtering approach.
#
# Returns:
# - similarities: for each product, a scored list of similar products.
# - item_counts: product popularity counts.
# - history: products each user has already interacted with.
def build_item_similarity(events: list[dict]) -> tuple[dict[str, Counter], Counter, dict[str, set[str]]]:
    # user_items[user_id][product_id] stores how strongly a user interacted
    # with a product.
    user_items: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    # Group all events by user and product.
    # If a user interacted with the same product multiple times, add the
    # weights together.
    for row in events:
        user_items[row["user_id"]][row["product_id"]] += interaction_weight(row)

    # item_counts counts how many users interacted with each product.
    item_counts = Counter()

    # co_counts[A][B] counts how often product A appears with product B in the
    # same user's history.
    co_counts: dict[str, Counter] = defaultdict(Counter)

    # Look at each user's product history.
    for items in user_items.values():
        product_ids = sorted(items)

        # Count each product once per user.
        for product_id in product_ids:
            item_counts[product_id] += 1

        # For every pair of products in the same user's history, add a
        # co-occurrence score. This means "people who used A also used B".
        for left, right in combinations(product_ids, 2):
            # Use the square root so very large interaction weights do not
            # dominate the similarity score too aggressively.
            weight = math.sqrt(items[left] * items[right])
            co_counts[left][right] += weight
            co_counts[right][left] += weight

    # Convert raw co-occurrence counts into normalized similarity scores.
    # This is similar to cosine similarity:
    # similarity(A, B) = co_count(A, B) / sqrt(count(A) * count(B))
    similarities: dict[str, Counter] = defaultdict(Counter)
    for product_id, neighbors in co_counts.items():
        for neighbor_id, co_count in neighbors.items():
            denominator = math.sqrt(item_counts[product_id] * item_counts[neighbor_id])
            if denominator:
                similarities[product_id][neighbor_id] = co_count / denominator

    # Keep only product IDs for user history. This helps us avoid recommending
    # products the user has already interacted with.
    history = {user_id: set(items) for user_id, items in user_items.items()}
    return similarities, item_counts, history


# Recommend products for one user.
#
# If the user has history:
# - Look at products they already interacted with.
# - Find similar products.
# - Sum similarity scores across all seen products.
#
# If the user has no history or no similar products:
# - Fall back to popular products.
def recommend_for_user(
    user_id: str,
    similarities: dict[str, Counter],
    popularity: Counter,
    history: dict[str, set[str]],
    limit: int,
) -> list[tuple[str, float]]:
    # Products the user already interacted with.
    seen = history.get(user_id, set())

    # Candidate recommendation scores.
    scores = Counter()

    # For every product the user has seen, add its similar products as
    # recommendation candidates.
    for product_id in seen:
        for neighbor_id, score in similarities.get(product_id, {}).items():
            # Do not recommend products the user already knows.
            if neighbor_id not in seen:
                scores[neighbor_id] += score

    # Cold-start fallback:
    # If we cannot personalize, recommend globally popular products.
    if not scores:
        for product_id, count in popularity.most_common():
            if product_id not in seen:
                scores[product_id] = float(count)

    # Return the highest scoring products.
    return scores.most_common(limit)


# Load data, build the similarity model, and print recommendations.
# This function is the user-facing "run recommender" step.
# It creates sample data automatically if no dataset exists yet.
def print_recommendations(user_id: str, limit: int) -> None:
    # If the user has not downloaded data or created sample data, create the
    # small sample dataset so the script still works.
    if not EVENTS_FILE.exists():
        print("No dataset found. Creating sample data first.")
        write_sample_data()

    # Load saved interactions and product metadata from disk.
    events = load_csv(EVENTS_FILE)
    products = load_products(PRODUCTS_FILE)

    # Build the in-memory item similarity model.
    similarities, popularity, history = build_item_similarity(events)

    # Generate recommendations for the requested user.
    recommendations = recommend_for_user(user_id, similarities, popularity, history, limit)

    print(f"\nRecommendations for user {user_id}:")

    # This happens when the requested user ID does not exist in the current
    # downloaded/sample dataset.
    if user_id not in history:
        print("User has no history in this dataset sample; using popularity fallback.")

    if not recommendations:
        print("No recommendations available.")
        return

    # Print a readable ranked list.
    for rank, (product_id, score) in enumerate(recommendations, start=1):
        product = products.get(product_id, {})
        title = product.get("title", product_id)
        category = product.get("category", "unknown")
        print(f"{rank}. {title} [{category}] product_id={product_id} score={score:.3f}")


# Define and parse command-line options.
#
# Examples:
# - `--sample` creates the tiny built-in dataset.
# - `--download --limit 5000` downloads a real Amazon review sample.
# - `--user-id <id>` chooses which user receives recommendations.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and run a small Amazon-style recommendation MVP.")

    # Data creation options.
    parser.add_argument("--download", action="store_true", help="Download a small Amazon reviews sample.")
    parser.add_argument("--sample", action="store_true", help="Write a tiny built-in sample dataset.")
    parser.add_argument("--limit", type=int, default=5000, help="Number of downloaded interactions.")

    # Recommendation options.
    parser.add_argument("--user-id", default="u1", help="User ID to recommend for.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations to print.")

    # Hugging Face dataset options. The direct file URL is used by default,
    # but these are available if you want to experiment with other datasets.
    parser.add_argument(
        "--dataset",
        default="McAuley-Lab/Amazon-Reviews-2023",
        help="Hugging Face dataset name.",
    )
    parser.add_argument("--config", default="raw_review_All_Beauty", help="Hugging Face dataset config.")
    parser.add_argument("--split", default="full", help="Hugging Face dataset split.")
    parser.add_argument(
        "--data-file-url",
        default=DEFAULT_DATA_FILE_URL,
        help="Direct JSONL/CSV/parquet file URL. Used by default to bypass deprecated dataset loaders.",
    )
    return parser.parse_args()


# Main entry point for the script.
#
# Execution order:
# 1. Read command-line arguments.
# 2. Optionally create sample data.
# 3. Optionally download real Amazon review data.
# 4. Build and print recommendations.
def main() -> None:
    args = parse_args()

    # Create local toy data when requested.
    if args.sample:
        write_sample_data()

    # Download real data when requested.
    if args.download:
        download_amazon_reviews(
            limit=args.limit,
            dataset=args.dataset,
            config=args.config,
            split=args.split,
            data_file_url=args.data_file_url,
        )

    # Always run recommendation after any requested data setup.
    print_recommendations(user_id=args.user_id, limit=args.top_k)

# Python only runs this block when the file is executed as a script:
# python -m src.amazon_recommendation_mvp
#
# This prevents `main()` from running if another Python file imports this file.

if __name__ == "__main__":
    main()

