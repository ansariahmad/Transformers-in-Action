import torch
import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Transformers in Action",
    page_icon="🏠",
)

st.sidebar.success("Select a Demo above.")

st.markdown(
    """
    # **Transformers in Action**  
    **Welcome to the Future of AI!**

    Discover the incredible power of modern **Transformer models** and how they can revolutionize the way you approach everyday tasks. Whether you want to analyze sentiment, fill in missing text, or classify data with zero-shot precision, this interactive app provides a seamless playground to explore Hugging Face models in action.

    ### **What Can You Do Here?**  
    🧠 **Sentiment Analysis** - Understand emotions in text, from happiness to frustration.  
    📝 **Fill Mask** - Predict missing words with precision using intelligent language models.  
    🚀 **Zero-Shot Classification** - Classify text into categories without pre-training.  
    ❓ **Question Answering** - Get instant answers to your queries with context-aware AI.  
    ✍️ **Text Summarization** - Condense lengthy content into concise summaries.  

    **Ready to experience the magic of AI?**  
    Pick a task from the left, explore, and bring your ideas to life!  

    """
)