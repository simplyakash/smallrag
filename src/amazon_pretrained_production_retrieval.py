import argparse
import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


# Production-style no-training recommendation service.
#
# This file avoids local model training. By default it uses a lightweight
# TF-IDF text encoder from scikit-learn plus FAISS retrieval, which fits small
# development environments. If you install the optional heavy requirements, it
# can also use a pretrained SentenceTransformers checkpoint.
#
# Architecture:
# - Data source: local Amazon sample CSV files.
# - Data validation: schema checks, with optional Great Expectations hook.
# - Feature generation: product text and user profile text.
# - "Product tower": text encoder embeds product text.
# - "User/session tower": same text encoder embeds user profile text.
# - Retrieval: FAISS searches nearest product embeddings.
# - Experiment metadata: optional MLflow logging.
# - Serving: FastAPI exposes /recommend and /health.
#
# This is not a trained Amazon two-tower model. It is a practical no-training
# production-style retrieval setup that demonstrates the same technologies and
# serving shape without needing an ID-specific checkpoint.

DATA_DIR = Path("data/amazon_recommendation")
EVENTS_FILE = DATA_DIR / "events.csv"
PRODUCTS_FILE = DATA_DIR / "products.csv"

ARTIFACT_DIR = Path("models/amazon_pretrained_retrieval")
FAISS_INDEX_FILE = ARTIFACT_DIR / "products.faiss"
PRODUCT_METADATA_FILE = ARTIFACT_DIR / "product_metadata.json"
BUILD_METADATA_FILE = ARTIFACT_DIR / "build_metadata.json"
TFIDF_VECTORIZER_FILE = ARTIFACT_DIR / "tfidf_vectorizer.pkl"
FEATURE_STORE_DIR = Path("models/amazon_feature_store")
PRODUCT_FEATURES_FILE = FEATURE_STORE_DIR / "product_features.json"
USER_FEATURES_FILE = FEATURE_STORE_DIR / "user_features.json"

DEFAULT_ENCODER = "tfidf"
HEAVY_SENTENCE_TRANSFORMER_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


# Request schema for the recommendation API.
# The user can recommend by known user_id or pass direct query text.
class RecommendationRequest(BaseModel):
    user_id: str | None = None
    query_text: str | None = None
    top_k: int = 10


# Read a CSV file into dictionaries.
def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


# Load product rows by product_id for fast lookup.
def load_products() -> dict[str, dict[str, str]]:
    if not PRODUCTS_FILE.exists():
        return {}
    return {row["product_id"]: row for row in load_csv(PRODUCTS_FILE)}


# Validate the minimum dataset contract before building embeddings.
#
# In a cloud production stack, this task would be handled by Great Expectations
# or TensorFlow Data Validation. Here we keep a small local contract so the
# pipeline fails early with a clear error.
def validate_data(events: list[dict[str, str]], products: dict[str, dict[str, str]]) -> None:
    if not events:
        raise ValueError(f"No events found in {EVENTS_FILE}")

    required_event_columns = {"user_id", "product_id", "event_type", "rating", "timestamp"}
    missing_event_columns = required_event_columns - set(events[0])
    if missing_event_columns:
        raise ValueError(f"events.csv missing columns: {sorted(missing_event_columns)}")

    if products:
        required_product_columns = {"product_id", "title", "category"}
        first_product = next(iter(products.values()))
        missing_product_columns = required_product_columns - set(first_product)
        if missing_product_columns:
            raise ValueError(f"products.csv missing columns: {sorted(missing_product_columns)}")


# Optional Great Expectations integration hook.
#
# This function intentionally keeps the local demo simple: it checks whether
# Great Expectations is installed and records that validation can be added here.
# A production version would create an expectation suite for schema, nulls,
# ranges, uniqueness, and drift checks.
def validate_with_great_expectations_if_available() -> str:
    try:
        import great_expectations as gx  # noqa: F401
    except ImportError:
        return "Great Expectations not installed; used built-in validation only."

    return "Great Expectations is installed; add expectation suites here for production validation."


# Build aggregate product statistics from user events.
# These become simple product features in the text sent to the pretrained model.
def compute_product_stats(events: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    counts = Counter()
    rating_sums = defaultdict(float)

    for row in events:
        product_id = row["product_id"]
        counts[product_id] += 1
        try:
            rating_sums[product_id] += float(row.get("rating") or 0)
        except ValueError:
            rating_sums[product_id] += 0

    stats = {}
    for product_id, count in counts.items():
        stats[product_id] = {
            "interaction_count": float(count),
            "average_rating": rating_sums[product_id] / max(count, 1),
        }
    return stats


# Convert product metadata and aggregate stats into natural language.
#
# A pretrained text model understands text, not raw database columns. This
# function is the feature engineering layer for the pretrained product tower.
def build_product_text(product: dict[str, str], stats: dict[str, float]) -> str:
    title = product.get("title") or product.get("product_id", "unknown product")
    category = product.get("category") or "unknown category"
    interaction_count = int(stats.get("interaction_count", 0))
    average_rating = stats.get("average_rating", 0.0)

    return (
        f"Product title: {title}. "
        f"Category: {category}. "
        f"Average rating: {average_rating:.2f}. "
        f"Historical interactions: {interaction_count}."
    )


# Build a text profile for a user from their interaction history.
#
# This acts as the no-training "user tower" input. The pretrained encoder turns
# this profile text into the user/session embedding used for retrieval.
def build_user_profile_text(
    user_id: str,
    events: list[dict[str, str]],
    products: dict[str, dict[str, str]],
    max_history: int = 20,
) -> str:
    user_events = [row for row in events if row["user_id"] == user_id]
    if not user_events:
        return "User is interested in popular beauty and personal care products."

    recent_events = user_events[-max_history:]
    product_descriptions = []

    for row in recent_events:
        product = products.get(row["product_id"], {})
        title = product.get("title", row["product_id"])
        category = product.get("category", "unknown")
        rating = row.get("rating", "")
        product_descriptions.append(f"{title} in {category} rated {rating}")

    return "User interacted with: " + "; ".join(product_descriptions)


# Load the text encoder.
#
# For `tfidf`, load the saved scikit-learn vectorizer.
# For `sentence-transformers/...`, load the optional pretrained checkpoint.
def load_encoder(model_name: str):
    if model_name == "tfidf":
        if not TFIDF_VECTORIZER_FILE.exists():
            raise FileNotFoundError(f"Missing {TFIDF_VECTORIZER_FILE}. Run with --build-index first.")
        with TFIDF_VECTORIZER_FILE.open("rb") as file:
            return pickle.load(file)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "SentenceTransformers is not installed. Either use --encoder tfidf "
            "or install optional heavy dependencies: "
            "python -m pip install -r requirements-production-recommendation-heavy.txt"
        ) from exc

    return SentenceTransformer(model_name)


# Encode text into normalized float32 vectors for FAISS.
#
# Normalization lets inner product search behave like cosine similarity.
def encode_texts(encoder: Any, texts: list[str], batch_size: int) -> np.ndarray:
    if hasattr(encoder, "transform"):
        vectors = encoder.transform(texts).astype("float32").toarray()
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    vectors = encoder.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return vectors.astype("float32")


# Build vectors with the selected no-training text encoder.
#
# `tfidf` is lightweight and works in constrained environments.
# SentenceTransformers is heavier but can be used on machines with enough disk.
def build_text_vectors(model_name: str, product_texts: list[str], batch_size: int) -> np.ndarray:
    if model_name == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer

        encoder = TfidfVectorizer(max_features=4096, stop_words="english")
        vectors = encoder.fit_transform(product_texts).astype("float32").toarray()
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)

        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        with TFIDF_VECTORIZER_FILE.open("wb") as file:
            pickle.dump(encoder, file)

        return vectors.astype("float32")

    encoder = load_encoder(model_name)
    return encode_texts(encoder, product_texts, batch_size=batch_size)


# Save JSON metadata with indentation so humans can inspect it.
def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


# Load JSON metadata from disk.
def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


# Load local feature store artifacts created by `src.amazon_feature_store`.
# In production this would be an online/offline feature store lookup through
# Feast, Tecton, or Vertex AI Feature Store.
def load_local_feature_store() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    product_features = load_json(PRODUCT_FEATURES_FILE) if PRODUCT_FEATURES_FILE.exists() else {}
    user_features = load_json(USER_FEATURES_FILE) if USER_FEATURES_FILE.exists() else {}
    return product_features, user_features


# Build product embeddings and a FAISS retrieval index.
#
# This is the no-training equivalent of building candidate generation artifacts
# in a production recommendation system.
def build_retrieval_artifacts(model_name: str, batch_size: int, log_mlflow: bool) -> None:
    import faiss

    if not EVENTS_FILE.exists():
        raise FileNotFoundError(f"Missing {EVENTS_FILE}. Run src.amazon_recommendation_mvp --download first.")

    events = load_csv(EVENTS_FILE)
    products = load_products()
    validate_data(events, products)
    validation_message = validate_with_great_expectations_if_available()

    feature_store_product_features, _ = load_local_feature_store()
    if feature_store_product_features:
        print(f"Loaded product features from local feature store: {PRODUCT_FEATURES_FILE}")
    else:
        print("Local feature store not found; computing product features directly from events.")

    product_stats = compute_product_stats(events)
    product_ids = sorted({row["product_id"] for row in events})
    product_metadata = []
    product_texts = []

    for product_id in product_ids:
        product = products.get(
            product_id,
            {"product_id": product_id, "title": product_id, "category": "unknown"},
        )
        feature_store_row = feature_store_product_features.get(product_id, {})
        stats = feature_store_row or product_stats.get(product_id, {})
        product = {**product, **feature_store_row}
        text = build_product_text(product, stats)
        product_metadata.append(
            {
                "product_id": product_id,
                "title": product.get("title", product_id),
                "category": product.get("category", "unknown"),
                "text": text,
            }
        )
        product_texts.append(text)

    product_vectors = build_text_vectors(model_name, product_texts, batch_size=batch_size)

    index = faiss.IndexFlatIP(product_vectors.shape[1])
    index.add(product_vectors)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_FILE))
    save_json(PRODUCT_METADATA_FILE, product_metadata)
    save_json(
        BUILD_METADATA_FILE,
        {
            "encoder": model_name,
            "num_products": len(product_metadata),
            "embedding_dim": int(product_vectors.shape[1]),
            "optional_heavy_encoder": HEAVY_SENTENCE_TRANSFORMER_ENCODER,
            "feature_store": str(FEATURE_STORE_DIR),
            "used_feature_store": bool(feature_store_product_features),
            "validation": validation_message,
        },
    )

    if log_mlflow:
        log_build_to_mlflow(model_name, len(product_metadata), int(product_vectors.shape[1]))

    print(f"Built FAISS index: {FAISS_INDEX_FILE}")
    print(f"Saved product metadata: {PRODUCT_METADATA_FILE}")
    print(validation_message)


# Optional MLflow logging.
#
# In production, this is where we would track build parameters, artifact
# versions, metrics, and eventually register model/index versions.
def log_build_to_mlflow(model_name: str, num_products: int, embedding_dim: int) -> None:
    try:
        import mlflow
    except ImportError:
        print("MLflow not installed; skipping experiment logging.")
        return

    mlflow.set_experiment("amazon_pretrained_retrieval")
    with mlflow.start_run(run_name="build_pretrained_faiss_index"):
        mlflow.log_param("encoder", model_name)
        mlflow.log_metric("num_products", num_products)
        mlflow.log_metric("embedding_dim", embedding_dim)
        mlflow.log_artifact(str(BUILD_METADATA_FILE))


# Load FAISS index and metadata artifacts.
def load_retrieval_artifacts():
    import faiss

    if not FAISS_INDEX_FILE.exists() or not PRODUCT_METADATA_FILE.exists():
        raise FileNotFoundError("Missing retrieval artifacts. Run with --build-index first.")

    index = faiss.read_index(str(FAISS_INDEX_FILE))
    product_metadata = load_json(PRODUCT_METADATA_FILE)
    build_metadata = load_json(BUILD_METADATA_FILE)
    return index, product_metadata, build_metadata


# Recommend products from either a user_id or direct query text.
def recommend(
    encoder: Any,
    index: Any,
    product_metadata: list[dict[str, Any]],
    events: list[dict[str, str]],
    products: dict[str, dict[str, str]],
    top_k: int,
    user_id: str | None = None,
    query_text: str | None = None,
) -> list[dict[str, Any]]:
    if query_text:
        profile_text = query_text
    elif user_id:
        _, user_features = load_local_feature_store()
        if user_id in user_features and user_features[user_id].get("recent_product_text"):
            profile_text = "User interacted with: " + user_features[user_id]["recent_product_text"]
        else:
            profile_text = build_user_profile_text(user_id, events, products)
    else:
        profile_text = "User is interested in popular useful shopping products."

    query_vector = encode_texts(encoder, [profile_text], batch_size=1)
    scores, indices = index.search(query_vector, top_k)

    results = []
    for score, product_index in zip(scores[0], indices[0]):
        item = dict(product_metadata[int(product_index)])
        item["score"] = float(score)
        results.append(item)

    return results


# Run one local recommendation from the command line.
def run_local_recommendation(model_name: str, user_id: str | None, query_text: str | None, top_k: int) -> None:
    index, product_metadata, build_metadata = load_retrieval_artifacts()
    events = load_csv(EVENTS_FILE)
    products = load_products()
    encoder = load_encoder(build_metadata.get("encoder", model_name))

    results = recommend(
        encoder=encoder,
        index=index,
        product_metadata=product_metadata,
        events=events,
        products=products,
        top_k=top_k,
        user_id=user_id,
        query_text=query_text,
    )

    print("\nPretrained retrieval recommendations:")
    for rank, item in enumerate(results, start=1):
        print(f"{rank}. {item['title']} [{item['category']}] product_id={item['product_id']} score={item['score']:.3f}")


# Create a production-style FastAPI app.
#
# Start it with:
# uvicorn src.amazon_pretrained_production_retrieval:create_app --factory --reload
def create_app() -> FastAPI:
    index, product_metadata, build_metadata = load_retrieval_artifacts()
    events = load_csv(EVENTS_FILE)
    products = load_products()
    encoder = load_encoder(build_metadata.get("encoder", DEFAULT_ENCODER))

    app = FastAPI(title="Amazon Pretrained Retrieval API")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "encoder": build_metadata.get("encoder"),
            "num_products": build_metadata.get("num_products"),
            "index": str(FAISS_INDEX_FILE),
        }

    @app.post("/recommend")
    def recommend_endpoint(request: RecommendationRequest) -> dict[str, Any]:
        results = recommend(
            encoder=encoder,
            index=index,
            product_metadata=product_metadata,
            events=events,
            products=products,
            top_k=request.top_k,
            user_id=request.user_id,
            query_text=request.query_text,
        )
        return {
            "user_id": request.user_id,
            "query_text": request.query_text,
            "recommendations": results,
        }

    return app


# Parse command-line options for local build and inference.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-training production-style Amazon retrieval.")
    parser.add_argument("--build-index", action="store_true", help="Build product embeddings and FAISS index.")
    parser.add_argument("--recommend", action="store_true", help="Run one local recommendation after artifacts exist.")
    parser.add_argument("--user-id", default=None, help="Known user ID from events.csv.")
    parser.add_argument("--query-text", default=None, help="Direct shopping intent text, such as 'hair care products'.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations.")
    parser.add_argument(
        "--encoder",
        default=DEFAULT_ENCODER,
        help="Use 'tfidf' for lightweight mode or a SentenceTransformers model name for optional heavy mode.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    parser.add_argument("--log-mlflow", action="store_true", help="Log artifact metadata to MLflow.")
    return parser.parse_args()


# CLI entry point.
def main() -> None:
    args = parse_args()

    if args.build_index:
        build_retrieval_artifacts(args.encoder, args.batch_size, args.log_mlflow)

    if args.recommend:
        run_local_recommendation(args.encoder, args.user_id, args.query_text, args.top_k)

    if not args.build_index and not args.recommend:
        print("Nothing to do. Use --build-index, --recommend, or both.")


if __name__ == "__main__":
    main()
