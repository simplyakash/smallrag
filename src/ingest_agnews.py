import os

import chromadb
from datasets import load_dataset

from src.rag_chroma_utils import build_embedding_function


AGNEWS_SAMPLE_SIZE = int(os.getenv("AGNEWS_SAMPLE_SIZE", "50"))
AGNEWS_BATCH_SIZE = int(os.getenv("AGNEWS_BATCH_SIZE", "10"))


def load_ag_news_dataset():
    split = f"train[:{AGNEWS_SAMPLE_SIZE}]"
    try:
        return load_dataset("ag_news", split=split)
    except Exception:
        return load_dataset("fancyzhx/ag_news", split=split)


# ----------------------------
# Load Real Dataset
# ----------------------------
dataset = load_ag_news_dataset()
print(f"Loaded {len(dataset)} documents from AG News dataset.")
print("Sample document:", dataset[0])

documents = [item["text"] for item in dataset]
ids = [f"doc_{i}" for i in range(len(documents))]

# ----------------------------
# Persistent DB Setup
# ----------------------------
chroma_client = chromadb.PersistentClient(path="../chroma_storage")

embedding_function = build_embedding_function()

collection = chroma_client.get_or_create_collection(
    name="agnews_collection",
    embedding_function=embedding_function
)

# ----------------------------
# Store Documents
# ----------------------------

for start in range(0, len(documents), AGNEWS_BATCH_SIZE):
    end = start + AGNEWS_BATCH_SIZE
    collection.upsert(
        documents=documents[start:end],
        ids=ids[start:end],
    )
    print(f"Stored documents {start + 1}-{min(end, len(documents))}.")

print("AG News documents stored successfully!")
