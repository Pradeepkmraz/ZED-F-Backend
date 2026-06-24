from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
# CORS is required so your frontend can communicate with the backend
CORS(app)

# Load your model components
try:
    with open("language_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        cv = pickle.load(f)
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    model, cv = None, None

@app.route("/", methods=["POST"])
def detect_language():
    if not model or not cv:
        return jsonify({"error": "Machine learning model not loaded on server."}), 500
        
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided."}), 400
        
    text = data["text"]
    if not text.strip():
        return jsonify({"error": "Please enter some text."}), 400
        
    try:
        # Vectorize and Predict
        vectorized_data = cv.transform([text]).toarray()
        prediction = model.predict(vectorized_data)
        
        return jsonify({"language": str(prediction[0])})
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    # Runs the API on port 5000 to match the HTML fetch URL
    app.run(port=5000, debug=True)