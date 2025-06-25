import os
import google.generativeai as genai
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

gen_model = genai.GenerativeModel("gemini-2.0-flash")

with open("context.txt", "r", encoding="utf-8") as f:
    full_context = f.read()

app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Thiếu câu hỏi"}), 400

    prompt = f"""Bạn là trợ lý AI. Hãy trả lời câu hỏi sau dựa vào ngữ cảnh dưới đây.

Ngữ cảnh:
{full_context}

Câu hỏi:
{question}

Trả lời bằng tiếng Việt:"""

    try:
        response = gen_model.generate_content(prompt)
        return jsonify({"question": question, "answer": response.text.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)