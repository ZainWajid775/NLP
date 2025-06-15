import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer
model = joblib.load("NLP/sentiment_model.pkl")
vectorizer = joblib.load("NLP/vectorizer.pkl")

# Introductory text
intro_text = """This app demonstrates the power and versatility of Natural Language Processing (NLP).  
You can explore multiple NLP tasks including:

- Sentiment Analysis  
- Text Summarization  
- Named Entity Recognition  
- Keyword Extraction"""

# Title and layout
st.markdown(
    "<h1 style='text-align: center; color:#FF3333;'>NLP Dashboard</h1>",
    unsafe_allow_html=True
)
st.write("")
st.text(intro_text)
st.write("")
st.write("")

# Input section
st.markdown("### Enter your review below:")
user_input = st.text_area("Type a review (movie, product, etc.)", height=150)
uploaded_file = st.file_uploader("Or upload a text file", type=["txt"])

text = ""
if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
elif user_input:
    text = user_input

if st.button("🔍 Analyze Sentiment"):
    if text.strip() == "":
        st.warning("Please enter or upload some text first.")
    else:
        with st.spinner("Analyzing..."):
            transformed_text = vectorizer.transform([text])
            prediction = model.predict(transformed_text)[0]
            proba = model.predict_proba(transformed_text)[0]

            label = "Positive" if prediction == 1 else "Negative"
            confidence = np.max(proba)

            st.markdown(
                f"""
                <div style='background-color:#111111; padding:15px; border-radius:10px; border-left:5px solid {"#33FF57" if prediction==1 else "#FF3355"}'>
                    <h4 style='color:{"#33FF57" if prediction==1 else "#FF3355"}'>Sentiment: {label}</h4>
                    <p style='color:white;'>Confidence: {confidence*100:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )