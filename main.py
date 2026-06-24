from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)  # This allows your custom HTML website to access the API safely

# Load your trained model and vectorizer
with open("language_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    cv = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    return """
    <html>
        <head><title>Language Detector API</title></head>
        <body style="font-family: Arial, sans-serif; text-align: center; padding-top: 100px; background-color: #f4f7f6;">
            <h1 style="color: #2c3e50;">🌐 Language Detection API is Online!</h1>
            <p style="color: #7f8c8d; font-size: 18px;">The backend is running perfectly on Render.</p>
            <p style="color: #e67e22; font-weight: bold;">Send POST requests to: <code>/predict</code></p>
        </body>
    </html>
    """


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the frontend request
        data = request.get_json()
        user_text = data.get('text', '')

        if not user_text.strip():
            return jsonify({'error': 'No text provided'}), 400

        # Transform and predict
        vectorized_data = cv.transform([user_text]).toarray()
        prediction = model.predict(vectorized_data)[0]

        # Return the language name back to the frontend
        return jsonify({'language': str(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
