import torch
import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Fill Mask",
    page_icon="📝")

st.write("# Fill Mask")
unmasker = pipeline('fill-mask', model='bert-base-uncased')

st.write("Enter a sentence with a masked word using `[MASK]`.")
user_input = st.text_input("Input your sentence:", "The capital of France is [MASK].")

num_responses = st.slider("Select the number of predictions:", min_value=1, max_value=20, value=5)

if st.button("Generate Predictions"):
    if "[MASK]" not in user_input:
        st.error("Please include '[MASK]' in your input sentence.")
    else:
        try:
            st.write("### Predictions:")
            predictions = unmasker(user_input, top_k=num_responses)
            for i, prediction in enumerate(predictions):
                token = prediction['token_str']
                score = prediction['score']
                user_input_before,user_input_after = user_input.split("[MASK]")
                user_input_with_token = user_input_before + "`" + token + "`"+ user_input_after
                st.write(user_input_with_token)
        except Exception as e:
            st.error(f"An error occurred: {e}")