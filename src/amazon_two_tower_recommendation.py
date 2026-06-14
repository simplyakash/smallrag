import argparse
import csv
import random
from pathlib import Path

import faiss
import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from torch import nn
from torch.utils.data import DataLoader, Dataset


# This file implements the "two-tower" recommendation approach described in
# the Amazon recommendation README.
#
# Technologies used:
# - PyTorch: trains the user tower and product tower neural networks.
# - FAISS: stores product embeddings and retrieves nearest products quickly.
# - FastAPI: exposes a small recommendation API when running in server mode.
#
# High-level flow:
# 1. Load user-product interactions from events.csv.
# 2. Convert string user/product IDs into integer indexes.
# 3. Train a two-tower model with positive and negative pairs.
# 4. Build a FAISS index from product embeddings.
# 5. Recommend nearest products for a user's embedding.

DATA_DIR = Path("data/amazon_recommendation")
EVENTS_FILE = DATA_DIR / "events.csv"
PRODUCTS_FILE = DATA_DIR / "products.csv"
MODEL_DIR = Path("models/amazon_two_tower")
MODEL_FILE = MODEL_DIR / "two_tower_model.pt"


# This small built-in dataset lets the script run even before the real Amazon
# sample has been downloaded.
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

SAMPLE_PRODUCTS = {
    "p1": {"product_id": "p1", "title": "Wireless Mouse", "category": "Electronics"},
    "p2": {"product_id": "p2", "title": "USB Keyboard", "category": "Electronics"},
    "p3": {"product_id": "p3", "title": "Running Shoes", "category": "Sports"},
    "p4": {"product_id": "p4", "title": "Laptop Stand", "category": "Office"},
    "p5": {"product_id": "p5", "title": "Water Bottle", "category": "Sports"},
}


# Create a directory when needed.
# The same helper is used for local data and model artifact folders.
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# Write a CSV file from rows represented as dictionaries.
# This is used only for creating the small fallback sample dataset.
def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# Create a tiny dataset if no downloaded Amazon data exists yet.
# This keeps the two-tower script runnable from a fresh checkout.
def write_sample_data() -> None:
    write_csv(EVENTS_FILE, SAMPLE_EVENTS, ["user_id", "product_id", "event_type", "rating", "timestamp"])
    write_csv(PRODUCTS_FILE, list(SAMPLE_PRODUCTS.values()), ["product_id", "title", "category"])
    print(f"Wrote sample interactions to {EVENTS_FILE}")
    print(f"Wrote sample products to {PRODUCTS_FILE}")


# Load a CSV file as a list of dictionaries.
# Each dictionary represents one row.
def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


# Load product metadata by product ID.
# The recommendation model only needs IDs, but metadata makes printed output
# easier to understand.
def load_products(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {row["product_id"]: row for row in load_csv(path)}


# Convert each raw interaction into a training strength.
# This lets purchases and high ratings count more than weak actions.
def interaction_weight(row: dict) -> float:
    event_weight = {
        "view": 1.0,
        "add_to_cart": 2.0,
        "purchase": 3.0,
    }.get(row.get("event_type", ""), 1.0)

    try:
        rating = float(row.get("rating") or 0)
    except ValueError:
        rating = 0

    return event_weight + max(rating - 3.0, 0.0)


# Read events and build integer ID mappings for PyTorch embedding layers.
#
# Neural embedding layers cannot directly use string IDs like "u1" or
# "B00YQ6X8EO", so we map:
# - user_id -> user_index
# - product_id -> product_index
def prepare_interactions(events: list[dict]) -> tuple[list[tuple[int, int, float]], dict[str, int], dict[str, int]]:
    user_to_idx: dict[str, int] = {}
    product_to_idx: dict[str, int] = {}
    interactions: list[tuple[int, int, float]] = []

    for row in events:
        user_id = row["user_id"]
        product_id = row["product_id"]

        if user_id not in user_to_idx:
            user_to_idx[user_id] = len(user_to_idx)
        if product_id not in product_to_idx:
            product_to_idx[product_id] = len(product_to_idx)

        interactions.append(
            (
                user_to_idx[user_id],
                product_to_idx[product_id],
                interaction_weight(row),
            )
        )

    return interactions, user_to_idx, product_to_idx


# PyTorch Dataset for two-tower training.
#
# For every positive pair (user, product), it samples a random negative product
# that the user interacted with less likely or not at all.
class TwoTowerDataset(Dataset):
    def __init__(self, interactions: list[tuple[int, int, float]], num_products: int):
        self.positive_pairs = interactions
        self.num_products = num_products

    def __len__(self) -> int:
        return len(self.positive_pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        user_idx, positive_product_idx, weight = self.positive_pairs[index]

        # Negative sampling teaches the model that a random product should score
        # lower than the product the user actually interacted with.
        negative_product_idx = random.randrange(self.num_products)
        while negative_product_idx == positive_product_idx and self.num_products > 1:
            negative_product_idx = random.randrange(self.num_products)

        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(positive_product_idx, dtype=torch.long),
            torch.tensor(negative_product_idx, dtype=torch.long),
            torch.tensor(weight, dtype=torch.float32),
        )


# Two-tower neural recommendation model.
#
# User tower:
# - Starts with a user embedding lookup.
# - Passes it through a small neural network.
#
# Product tower:
# - Starts with a product embedding lookup.
# - Passes it through a separate small neural network.
#
# The final score is a dot product between normalized user and product vectors.
class TwoTowerModel(nn.Module):
    def __init__(self, num_users: int, num_products: int, embedding_dim: int):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.product_embedding = nn.Embedding(num_products, embedding_dim)

        self.user_tower = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.product_tower = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def encode_users(self, user_indices: torch.Tensor) -> torch.Tensor:
        user_vectors = self.user_embedding(user_indices)
        user_vectors = self.user_tower(user_vectors)
        return nn.functional.normalize(user_vectors, dim=-1)

    def encode_products(self, product_indices: torch.Tensor) -> torch.Tensor:
        product_vectors = self.product_embedding(product_indices)
        product_vectors = self.product_tower(product_vectors)
        return nn.functional.normalize(product_vectors, dim=-1)

    def score(self, user_indices: torch.Tensor, product_indices: torch.Tensor) -> torch.Tensor:
        user_vectors = self.encode_users(user_indices)
        product_vectors = self.encode_products(product_indices)
        return (user_vectors * product_vectors).sum(dim=-1)


# Train the two-tower model with pairwise ranking loss.
#
# For each user:
# - positive product should score high
# - negative random product should score low
#
# The loss encourages:
# score(user, positive_product) > score(user, negative_product)
def train_model(
    interactions: list[tuple[int, int, float]],
    num_users: int,
    num_products: int,
    embedding_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> TwoTowerModel:
    model = TwoTowerModel(num_users=num_users, num_products=num_products, embedding_dim=embedding_dim)
    dataset = TwoTowerDataset(interactions, num_products=num_products)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for user_idx, positive_idx, negative_idx, weight in loader:
            positive_score = model.score(user_idx, positive_idx)
            negative_score = model.score(user_idx, negative_idx)

            # Softplus ranking loss:
            # If positive_score is already greater than negative_score, loss is
            # small. If not, the loss is large.
            loss = nn.functional.softplus(negative_score - positive_score)
            loss = (loss * weight).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        average_loss = total_loss / max(len(loader), 1)
        print(f"Epoch {epoch}/{epochs} loss={average_loss:.4f}")

    return model


# Build a FAISS index from product embeddings.
#
# FAISS is the retrieval engine. Instead of comparing a user vector against
# every product with slow Python loops, FAISS performs fast vector search.
def build_faiss_index(model: TwoTowerModel, num_products: int) -> tuple[faiss.IndexFlatIP, np.ndarray]:
    model.eval()
    with torch.no_grad():
        product_indices = torch.arange(num_products, dtype=torch.long)
        product_vectors = model.encode_products(product_indices).cpu().numpy().astype("float32")

    # IndexFlatIP uses inner product. Because vectors are normalized, inner
    # product behaves like cosine similarity.
    index = faiss.IndexFlatIP(product_vectors.shape[1])
    index.add(product_vectors)
    return index, product_vectors


# Save the trained model and ID mappings to disk.
# These artifacts let us load the model later for serving or inference.
def save_artifacts(
    model: TwoTowerModel,
    user_to_idx: dict[str, int],
    product_to_idx: dict[str, int],
    embedding_dim: int,
) -> None:
    ensure_dir(MODEL_DIR)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "user_to_idx": user_to_idx,
            "product_to_idx": product_to_idx,
            "embedding_dim": embedding_dim,
        },
        MODEL_FILE,
    )
    print(f"Saved model artifacts to {MODEL_FILE}")


# Load model artifacts from disk.
# This is used by server mode so the API can start without retraining.
# The checkpoint may come from real training (`--train`) or from architecture
# demo mode (`--demo-no-train`).
def load_artifacts() -> tuple[TwoTowerModel, dict[str, int], dict[str, int]]:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}. Run with --train or --demo-no-train first.")

    checkpoint = torch.load(MODEL_FILE, map_location="cpu")
    user_to_idx = checkpoint["user_to_idx"]
    product_to_idx = checkpoint["product_to_idx"]
    embedding_dim = checkpoint["embedding_dim"]

    model = TwoTowerModel(
        num_users=len(user_to_idx),
        num_products=len(product_to_idx),
        embedding_dim=embedding_dim,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, user_to_idx, product_to_idx


# Recommend products for a user using the two-tower model and FAISS index.
#
# Steps:
# 1. Convert user_id to integer user index.
# 2. Generate user embedding with the user tower.
# 3. Search nearest product embeddings with FAISS.
# 4. Convert product indexes back to product IDs and titles.
def recommend_for_user(
    user_id: str,
    model: TwoTowerModel,
    index: faiss.IndexFlatIP,
    user_to_idx: dict[str, int],
    product_to_idx: dict[str, int],
    products: dict[str, dict],
    top_k: int,
) -> list[dict]:
    if user_id not in user_to_idx:
        raise ValueError(f"Unknown user_id: {user_id}")

    idx_to_product = {idx: product_id for product_id, idx in product_to_idx.items()}
    user_idx = torch.tensor([user_to_idx[user_id]], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        user_vector = model.encode_users(user_idx).cpu().numpy().astype("float32")

    scores, indices = index.search(user_vector, top_k)

    recommendations = []
    for score, product_idx in zip(scores[0], indices[0]):
        product_id = idx_to_product[int(product_idx)]
        product = products.get(product_id, {})
        recommendations.append(
            {
                "product_id": product_id,
                "title": product.get("title", product_id),
                "category": product.get("category", "unknown"),
                "score": float(score),
            }
        )

    return recommendations


# Print recommendations in a readable ranked list.
# Keeping this in one helper lets train mode and no-train demo mode display
# results in the same format.
def print_ranked_recommendations(user_id: str, recommendations: list[dict], title: str) -> None:
    print(f"\n{title} for user {user_id}:")
    for rank, item in enumerate(recommendations, start=1):
        print(
            f"{rank}. {item['title']} [{item['category']}] "
            f"product_id={item['product_id']} score={item['score']:.3f}"
        )


# Train the model from local CSV files and print recommendations.
# If events.csv does not exist, this function creates the tiny sample dataset.
def train_and_recommend(args: argparse.Namespace) -> None:
    if not EVENTS_FILE.exists():
        print("No dataset found. Creating sample data first.")
        write_sample_data()

    events = load_csv(EVENTS_FILE)
    products = load_products(PRODUCTS_FILE)
    interactions, user_to_idx, product_to_idx = prepare_interactions(events)

    print(f"Loaded {len(interactions)} interactions.")
    print(f"Users: {len(user_to_idx)}")
    print(f"Products: {len(product_to_idx)}")

    model = train_model(
        interactions=interactions,
        num_users=len(user_to_idx),
        num_products=len(product_to_idx),
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    save_artifacts(model, user_to_idx, product_to_idx, args.embedding_dim)

    index, _ = build_faiss_index(model, num_products=len(product_to_idx))

    user_id = args.user_id
    if user_id not in user_to_idx:
        user_id = next(iter(user_to_idx))
        print(f"Requested user was not found. Using sample user_id={user_id}")

    recommendations = recommend_for_user(
        user_id=user_id,
        model=model,
        index=index,
        user_to_idx=user_to_idx,
        product_to_idx=product_to_idx,
        products=products,
        top_k=args.top_k,
    )

    print_ranked_recommendations(user_id, recommendations, "Trained two-tower recommendations")


# Run the two-tower architecture without training.
#
# This is useful for learning the technology flow:
# - PyTorch still creates user/product tower networks.
# - FAISS still indexes product embeddings.
# - Recommendation still searches nearest products for a user embedding.
#
# Important:
# The recommendations are not meaningful because the weights are random. This
# mode is for understanding architecture and integration, not model quality.
def demo_without_training(args: argparse.Namespace) -> None:
    if not EVENTS_FILE.exists():
        print("No dataset found. Creating sample data first.")
        write_sample_data()

    events = load_csv(EVENTS_FILE)
    products = load_products(PRODUCTS_FILE)
    interactions, user_to_idx, product_to_idx = prepare_interactions(events)

    print(f"Loaded {len(interactions)} interactions.")
    print(f"Users: {len(user_to_idx)}")
    print(f"Products: {len(product_to_idx)}")
    print("Creating a deterministic untrained two-tower model for architecture demo.")

    # Fixed seed makes the random demo checkpoint repeatable across runs.
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model = TwoTowerModel(
        num_users=len(user_to_idx),
        num_products=len(product_to_idx),
        embedding_dim=args.embedding_dim,
    )
    save_artifacts(model, user_to_idx, product_to_idx, args.embedding_dim)

    index, _ = build_faiss_index(model, num_products=len(product_to_idx))

    user_id = args.user_id
    if user_id not in user_to_idx:
        user_id = next(iter(user_to_idx))
        print(f"Requested user was not found. Using sample user_id={user_id}")

    recommendations = recommend_for_user(
        user_id=user_id,
        model=model,
        index=index,
        user_to_idx=user_to_idx,
        product_to_idx=product_to_idx,
        products=products,
        top_k=args.top_k,
    )

    print_ranked_recommendations(user_id, recommendations, "Untrained demo two-tower recommendations")
    print("\nNote: scores are from random weights. Use --train later for meaningful recommendations.")


# Request body for FastAPI recommendation calls.
class RecommendationRequest(BaseModel):
    user_id: str
    top_k: int = 10


# Create a FastAPI app backed by the saved model.
# Run after `--train` or `--demo-no-train`:
# uvicorn src.amazon_two_tower_recommendation:create_app --factory --reload
def create_app() -> FastAPI:
    model, user_to_idx, product_to_idx = load_artifacts()
    products = load_products(PRODUCTS_FILE)
    index, _ = build_faiss_index(model, num_products=len(product_to_idx))

    app = FastAPI(title="Amazon Two-Tower Recommendation API")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": str(MODEL_FILE)}

    @app.post("/recommend")
    def recommend(request: RecommendationRequest) -> dict:
        recommendations = recommend_for_user(
            user_id=request.user_id,
            model=model,
            index=index,
            user_to_idx=user_to_idx,
            product_to_idx=product_to_idx,
            products=products,
            top_k=request.top_k,
        )
        return {"user_id": request.user_id, "recommendations": recommendations}

    return app


# Parse command-line arguments for training and local inference.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a PyTorch two-tower Amazon recommendation MVP.")
    parser.add_argument("--train", action="store_true", help="Train the two-tower model and save artifacts.")
    parser.add_argument(
        "--demo-no-train",
        action="store_true",
        help="Create an untrained demo checkpoint to understand the architecture without training.",
    )
    parser.add_argument("--user-id", default="u1", help="User ID to recommend for.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations to print.")
    parser.add_argument("--embedding-dim", type=int, default=32, help="Embedding vector size.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.005, help="Adam optimizer learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for no-training demo mode.")
    return parser.parse_args()


# Script entry point.
# This CLI can either train a model or create an untrained architecture demo.
def main() -> None:
    args = parse_args()

    if args.demo_no_train:
        demo_without_training(args)
        return

    if args.train:
        train_and_recommend(args)
        return

    print("Nothing to do. Run with --demo-no-train for architecture demo or --train for real training.")


if __name__ == "__main__":
    main()
