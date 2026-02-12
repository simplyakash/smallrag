import chromadb
from chromadb.utils import embedding_functions
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
# ----------------------------
# Connect to persistent DB
# ----------------------------
chroma_client = chromadb.PersistentClient(path="../chroma_storage")

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

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

# ----------------------------
# Local LLM (Flan-T5)
# ----------------------------
# generator = pipeline(
#     "text2text-generation",
#     model="google/flan-t5-base",
#     device=0  # use -1 if CPU
# )

#generator = pipeline("text-generation", model="google/flan-t5-base")

prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{query}
"""
print("Prompt for LLM: \n")
print(prompt)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#print(tokenizer.decode(outputs[0], skip_special_tokens=True))
#response = generator(prompt, max_new_tokens=200)

print("\nAnswer: response: \n")
print(response)