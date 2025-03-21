from AI.create_data import create_data_chroma_db
from AI.search_data import search_data_chroma_db
from AI.mood_analyzer import check_mood

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/api/v1/create', methods=['POST'])
def api_create():
    create_data_chroma_db("C:\\Users\\ASUS\\PycharmProjects\\AIDemo\\info.txt", "./market_chroma_db")
    create_response = {"result": "Success!"}

    return jsonify(create_response)


@app.route('/api/v1/search', methods=['POST'])
def api_search():
    data = request.get_json()

    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    # Получаем текст из запроса
    question = data['text']

    answer = search_data_chroma_db(question, "./market_chroma_db")

    answer_response = {"answer": answer}

    return jsonify(answer_response)


@app.route('/api/v1/mood', methods=['POST'])
def api_mood():
    data = request.get_json()

    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    # Получаем текст из запроса
    question = data['text']

    answer = check_mood(question)

    answer_response = {"answer": answer}

    return jsonify(answer_response)


if __name__ == '__main__':
    app.run(debug=True)
