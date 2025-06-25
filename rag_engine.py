import os
import json
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Không tìm thấy GEMINI_API_KEY trong file .env!")

genai.configure(api_key=api_key)

DATA_DIR = "data"
CONTEXTS_PATH = os.path.join(DATA_DIR, "contexts.json")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")

if not os.path.exists(CONTEXTS_PATH) or not os.path.exists(INDEX_PATH):
    print("⚠️ Thiếu contexts hoặc index, đang tạo lại từ build_index.py...")
    os.system("python build_index.py")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
gen_model = genai.GenerativeModel("gemini-2.0-flash")

with open(CONTEXTS_PATH, "r", encoding="utf-8") as f:
    contexts = json.load(f)

context_texts = [item["text"] for item in contexts]
context_embeddings = [item["embedding"] for item in contexts]

dimension = len(context_embeddings[0])
index = faiss.read_index(INDEX_PATH)

def retrieve_context(question, top_k=3):
    q_emb = embed_model.encode([question])
    D, I = index.search(np.array(q_emb).astype("float32"), top_k)
    return [context_texts[i] for i in I[0]]

def generate_answer(question):
    contexts = retrieve_context(question, top_k=3)
    combined_context = "\n\n".join(contexts)

    prompt = f"""Bạn là trợ lý AI. Hãy trả lời câu hỏi dưới đây dựa trên ngữ cảnh.

Ngữ cảnh:
{combined_context}

Câu hỏi:
{question}

Trả lời bằng tiếng Việt:"""

    try:
        response = gen_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Lỗi từ mô hình Gemini: {e}"
