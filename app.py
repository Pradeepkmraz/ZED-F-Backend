import streamlit as st
import pickle
import numpy as np

# 1. Web Page Title & Styling
st.set_page_config(page_title="Language Detector", page_icon="🌐", layout="centered")
st.title("🌐 Language Detection Model")
st.write("Type or paste any sentence below to detect its language instantly.")

# 2. Function to safely load your saved pickle models
@st.cache_resource
def load_saved_assets():
    with open("language_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        cv = pickle.load(f)
    return model, cv

# Load the model components
try:
    model, cv = load_saved_assets()
except FileNotFoundError:
    st.error("Error: 'language_model.pkl' or 'vectorizer.pkl' not found. Make sure you uploaded them to GitHub!")

# 3. Create Interactive Frontend Web Elements
user_input = st.text_area("Enter text here:", placeholder="e.g., My name is Pradeep Yadav")

# 4. Process user input when button is clicked
if st.button("Predict Language"):
    if not user_input.strip():
        st.warning("Please enter some text before predicting.")
    else:
        # Convert user text into numbers using the loaded vectorizer
        vectorized_data = cv.transform([user_input]).toarray()
        
        # Predict language using the loaded model
        prediction = model.predict(vectorized_data)
        
        # Output result cleanly
        st.success(f"🎉 Predicted Language: **{prediction[0]}**")
