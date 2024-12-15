import streamlit as st
from transformers import pipeline


st.set_page_config(
    page_title="Question Answer",
    page_icon="✍️")

st.write("# Text Summarization")

# Model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

user_input = st.text_area("Enter text to summarize")

if st.button("Generate Predictions"):
        try:
            st.write("## Summary:")
            generated_summary = summarizer(user_input)
            st.write(generated_summary[0]["summary_text"])
        except Exception as e:
            st.error(f"An error occurred: {e}")