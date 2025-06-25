import os
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
from datasets import load_dataset
from difflib import SequenceMatcher

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

gen_model = genai.GenerativeModel("gemini-2.0-flash")

dataset = load_dataset("json", data_files={"train": "data/data.gzip"}, split="train")
contexts = dataset["context"]
questions = dataset["question"]

def retrieve_best_context(question, top_k=3):
    scores = [
        (SequenceMatcher(None, question.lower(), q.lower()).ratio(), i)
        for i, q in enumerate(questions)
    ]
    scores.sort(reverse=True)
    best_indices = [i for _, i in scores[:top_k]]
    return [contexts[i] for i in best_indices]

app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Thiếu câu hỏi"}), 400

    try:
        best_contexts = retrieve_best_context(question, top_k=3)
        combined_context = "\n\n".join(best_contexts)

        prompt = f"""Bạn là trợ lý AI. Hãy trả lời câu hỏi dưới đây dựa vào ngữ cảnh.

Ngữ cảnh:
{combined_context}

Câu hỏi:
{question}

Trả lời bằng tiếng Việt:"""

        response = gen_model.generate_content(prompt)
        return jsonify({"question": question, "answer": response.text.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)