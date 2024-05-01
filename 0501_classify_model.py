import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# loading model
model = pickle.load(open("text_classification_model.pkl", "rb"))

def classify_news(text):
    prediction = model.predict([text])
    return prediction[0]

def main():
    st.title("News Classify Model")
    st.write("Classify news and predict its categories")
    
    news_text = st.text_area("Input news", "")
    
    if st.button("Classify"):
        label = classify_news(news_text)
        st.write(f"The category of this news is: {label}")

if __name__ == "__main__":
    main()
