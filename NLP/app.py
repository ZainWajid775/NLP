import streamlit as st
import joblib
import os


# load saved model
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

intro_text = """This app demonstrates the power and versatility of Natural Language Processing (NLP).  
You can explore multiple NLP tasks including:

- Sentiment Analysis  
- Text Summarization  
- Named Entity Recognition  
- Keyword Extraction  """

st.markdown(
    """
    <h1 style='text-align: center; '>NLP Dashboard</h1>
    """,
    unsafe_allow_html=True
)

st.write("")               
st.text(intro_text)
st.write("")
st.write("")

st.markdown("### Enter your review below:")
# User input manual
user_input = st.text_area("Type a review (movie, product, etc.)", height=150)
# Text File input
uploaded_file = st.file_uploader("Or upload a text file", type=["txt"])
text = ""


if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
elif user_input:
    text = user_input


# Confirm button
if st.button("🔍 Analyze Sentiment"):
    if text.strip() == "":
        st.warning("Please enter or upload some text first.")
    else:
        with st.spinner("Analyzing..."):
            # Preprocess and predict
            transformed_text = vectorizer.transform([text])
            prediction = model.predict(transformed_text)[0]
            if prediction == 1:
                st.success("Sentiment: Positive")
            else:
                st.error("Sentiment: Negative")


