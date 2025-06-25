from flask import Flask, request, jsonify
from rag_engine import generate_answer

app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Thiếu câu hỏi"}), 400
    
    answer = generate_answer(question)
    return jsonify({"question": question, "answer": answer})

if __name__ == "__main__":
    app.run(debug=True)