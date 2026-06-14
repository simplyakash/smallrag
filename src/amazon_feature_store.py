import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path


# Local production-style feature store.
#
# In a real production stack this role would usually be handled by Feast,
# Tecton, or Vertex AI Feature Store. For this repo, we materialize features
# into local JSON artifacts so the rest of the recommendation service can
# actually run without cloud accounts or a separate online store.

DATA_DIR = Path("data/amazon_recommendation")
EVENTS_FILE = DATA_DIR / "events.csv"
PRODUCTS_FILE = DATA_DIR / "products.csv"

FEATURE_STORE_DIR = Path("models/amazon_feature_store")
PRODUCT_FEATURES_FILE = FEATURE_STORE_DIR / "product_features.json"
USER_FEATURES_FILE = FEATURE_STORE_DIR / "user_features.json"
FEATURE_METADATA_FILE = FEATURE_STORE_DIR / "feature_metadata.json"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_products() -> dict[str, dict[str, str]]:
    if not PRODUCTS_FILE.exists():
        return {}
    return {row["product_id"]: row for row in load_csv(PRODUCTS_FILE)}


def parse_rating(value: str | None) -> float:
    try:
        return float(value or 0)
    except ValueError:
        return 0.0


def materialize_product_features(events: list[dict[str, str]], products: dict[str, dict[str, str]]) -> dict[str, dict]:
    counts = Counter()
    rating_sums = defaultdict(float)
    user_counts = defaultdict(set)

    for row in events:
        product_id = row["product_id"]
        counts[product_id] += 1
        rating_sums[product_id] += parse_rating(row.get("rating"))
        user_counts[product_id].add(row["user_id"])

    product_features = {}
    for product_id, count in counts.items():
        product = products.get(product_id, {})
        product_features[product_id] = {
            "product_id": product_id,
            "title": product.get("title", product_id),
            "category": product.get("category", "unknown"),
            "interaction_count": count,
            "unique_user_count": len(user_counts[product_id]),
            "average_rating": rating_sums[product_id] / max(count, 1),
        }

    return product_features


def materialize_user_features(events: list[dict[str, str]], products: dict[str, dict[str, str]]) -> dict[str, dict]:
    user_events = defaultdict(list)

    for row in events:
        user_events[row["user_id"]].append(row)

    user_features = {}
    for user_id, rows in user_events.items():
        ratings = [parse_rating(row.get("rating")) for row in rows]
        recent_rows = rows[-20:]
        recent_product_ids = [row["product_id"] for row in recent_rows]
        recent_product_text = []

        for row in recent_rows:
            product = products.get(row["product_id"], {})
            title = product.get("title", row["product_id"])
            category = product.get("category", "unknown")
            recent_product_text.append(f"{title} in {category} rated {row.get('rating', '')}")

        user_features[user_id] = {
            "user_id": user_id,
            "interaction_count": len(rows),
            "average_rating": sum(ratings) / max(len(ratings), 1),
            "recent_product_ids": recent_product_ids,
            "recent_product_text": "; ".join(recent_product_text),
        }

    return user_features


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def materialize_features() -> None:
    if not EVENTS_FILE.exists():
        raise FileNotFoundError(f"Missing {EVENTS_FILE}. Run src.amazon_recommendation_mvp --download first.")

    events = load_csv(EVENTS_FILE)
    products = load_products()
    product_features = materialize_product_features(events, products)
    user_features = materialize_user_features(events, products)

    metadata = {
        "materialized_at": datetime.now(UTC).isoformat(),
        "source_events": str(EVENTS_FILE),
        "source_products": str(PRODUCTS_FILE),
        "product_feature_count": len(product_features),
        "user_feature_count": len(user_features),
        "production_mapping": {
            "local": str(FEATURE_STORE_DIR),
            "feast": "feature_store/recommendation_feature_repo/",
            "tecton": "managed online/offline feature platform",
            "vertex_ai_feature_store": "managed GCP feature store",
        },
    }

    save_json(PRODUCT_FEATURES_FILE, product_features)
    save_json(USER_FEATURES_FILE, user_features)
    save_json(FEATURE_METADATA_FILE, metadata)

    print(f"Materialized product features: {PRODUCT_FEATURES_FILE}")
    print(f"Materialized user features: {USER_FEATURES_FILE}")
    print(f"Feature metadata: {FEATURE_METADATA_FILE}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize local recommendation feature store artifacts.")
    parser.add_argument("--materialize", action="store_true", help="Build local product/user feature store artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.materialize:
        materialize_features()
    else:
        print("Nothing to do. Use --materialize.")


if __name__ == "__main__":
    main()
