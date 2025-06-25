from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os

dataset_path = "data/data.gzip"
dataset = load_dataset("json", data_files={"train": dataset_path}, split="train")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

contexts = []
embeddings = []

for i, row in enumerate(dataset):
    ctx = row["context"].strip()
    emb = embed_model.encode(ctx)
    
    contexts.append({
        "id": i,
        "text": ctx,
        "embedding": emb.tolist()
    })
    embeddings.append(emb)

emb_matrix = np.array(embeddings).astype("float32")
index = faiss.IndexFlatL2(emb_matrix.shape[1])
index.add(emb_matrix)

os.makedirs("data", exist_ok=True)
faiss.write_index(index, "data/faiss_index.bin")

with open("data/contexts.json", "w", encoding="utf-8") as f:
    json.dump(contexts, f, ensure_ascii=False, indent=2)

print("✅ Đã tạo: contexts.json và faiss_index.bin.")
