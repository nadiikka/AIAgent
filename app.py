from flask import Flask, request, render_template, jsonify
import pickle
import ai_response as ai

app = Flask(__name__)

health_model = pickle.load(open('models/health.pkl', 'rb'))
subject_model = pickle.load(open('models/subject.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html', data='index')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = str(data.get('text', ''))

    if not data or not text:
        return jsonify({'error': 'No data provided'}), 400

    health = str(health_model.predict([text])[0])
    subject = str(subject_model.predict([text])[0])

    response = ai.get_chat_response(text, health, subject)

    result = {
        'response': response,
    }

    return jsonify(result), 200

if __name__ == '__main__':
    app.run(debug=True)