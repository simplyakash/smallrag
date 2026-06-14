import chromadb

from src.rag_chroma_utils import build_embedding_function
# ----------------------------
# Connect to persistent DB
# ----------------------------
chroma_client = chromadb.PersistentClient(path="../chroma_storage")

embedding_function = build_embedding_function()

collection = chroma_client.get_collection(
    name="agnews_collection",
    embedding_function=embedding_function
)

# ----------------------------
# Retrieval
# ----------------------------
query = "Tell me about business news related to stocks."

results = collection.query(
    query_texts=[query],
    n_results=3
)

context = "\n\n".join(results["documents"][0])

prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{query}
"""
print("Prompt for LLM: \n")
print(prompt)
print("\nRetrieved context: \n")
print(context)