import streamlit as st
from transformers import pipeline


st.set_page_config(
    page_title="Question Answer",
    page_icon="❓")

# App Name
st.write("# Question Answer")

# Model
qa_model = pipeline("question-answering", model="distilbert/distilbert-base-cased-distilled-squad")


st.write("Provide context and question.")

question = st.text_input("Enter your question:")
context = st.text_input("Enter the context:")

if st.button("Generate Answer"):
    if not (question or context):
        st.warning("Provide both question and context.")
    else:
        try:
            st.write("## Answer")
            ans = qa_model(question=question, context=context)
            st.write(ans['answer'])
        except Exception as e:
            st.error(f"An error occurred: {e}")
            