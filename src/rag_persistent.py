import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

# ----------------------------
# 1. Persistent DB Setup
# ----------------------------
chroma_client = chromadb.PersistentClient(path="./chroma_storage")

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    name="company_docs",
    embedding_function=embedding_function
)

# ----------------------------
# 2. Add Documents (run once)
# ----------------------------
documents = [
    "Employees are entitled to 20 days of paid leave per year.",
    "The company follows a hybrid work policy.",
    "Engineering teams use GitHub for version control.",
    "Security policies require 2FA for all internal tools."
]

ids = [f"id_{i}" for i in range(len(documents))]

collection.add(
    documents=documents,
    ids=ids
)

print("Documents stored successfully!")
