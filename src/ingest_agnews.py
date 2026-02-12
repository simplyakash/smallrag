import chromadb
from chromadb.utils import embedding_functions
from datasets import load_dataset

# ----------------------------
# Load Real Dataset
# ----------------------------
dataset = load_dataset("ag_news", split="train[:500]")  # first 500 sample\s
print(f"Loaded {len(dataset)} documents from AG News dataset.")
print("Sample document:", dataset[0])

documents = [item["text"] for item in dataset]
ids = [f"doc_{i}" for i in range(len(documents))]

# ----------------------------
# Persistent DB Setup
# ----------------------------
chroma_client = chromadb.PersistentClient(path="../chroma_storage")

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    name="agnews_collection",
    embedding_function=embedding_function
)

# ----------------------------
# Store Documents
# ----------------------------

collection.add(
    documents=documents,
    ids=ids
)

print("AG News documents stored successfully!")
