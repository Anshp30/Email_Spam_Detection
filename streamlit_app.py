import streamlit as st
import pickle

# Load model & vectorizer
import pickle
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "spam_model.pkl")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Email Spam Detection", layout="centered")

st.title("📧 Email Spam Detection System")
st.write("Check whether an email message is **Spam or Not Spam**")

# Input box
message = st.text_area("Enter email text here:")

# Predict button
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
