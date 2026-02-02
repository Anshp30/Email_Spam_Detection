import streamlit as st
import pickle
import os

# -------------------------------
# Load model and vectorizer safely
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "spam_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

# Check if files exist before loading
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}. Please run model.py first to generate the model.")
    st.stop()

if not os.path.exists(VECTORIZER_PATH):
    st.error(f"Vectorizer file not found at {VECTORIZER_PATH}. Please run model.py first to generate the vectorizer.")
    st.stop()

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model files: {str(e)}")
    st.stop()

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Email Spam Detection", layout="centered")

st.title("📧 Email Spam Detection System")
st.write("Check whether an email message is **Spam or Not Spam**")

message = st.text_area("Enter email text here:")

if st.button("Check Spam"):
    if message.strip() == "":
        st.warning("Please enter a message")
    else:
        data = vectorizer.transform([message])
        prediction = model.predict(data)

        if prediction[0] == 1:
            st.error("❌ This is a SPAM message")
        else:
            st.success("✅ This is NOT a spam message")
