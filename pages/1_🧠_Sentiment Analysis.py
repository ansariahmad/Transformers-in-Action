import torch
import numpy as np
import streamlit as st
from torch.nn import Softmax
import plotly.graph_objects as go
from transformers import AutoConfig, AutoTokenizer
from transformers import AutoModelForSequenceClassification


st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🧠")

st.write("# Sentiment Analysis")


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

user_input = st.text_input('What\'s in your mind?')

if st.button("Perform Sentiment Analysis"):
    if not user_input:
        st.warning("Please enter some text!")
    else:
        try:
            st.write("## Sentiment Plot")
            encoded_input = tokenizer(user_input, return_tensors='pt')
            output = model(**encoded_input)
            scores = output[0][0].detach().numpy()
            softmax = Softmax(dim=1)
            scores = softmax(torch.tensor([scores]))
            scores = scores.numpy()[0]

            categories = []
            probabilities = []
            ranking = np.argsort(scores)
            ranking = ranking[::-1]
            for i in range(scores.shape[0]):
                categories.append(config.id2label[ranking[i]])
                probabilities.append(np.round(float(scores[ranking[i]]), 4).tolist())

            res = [[cat, sco] for cat,sco in zip(categories, probabilities)]
            res.sort(key=lambda x: x[0], reverse=True)
            probabilities = [i[1] for i in res]


            # Create the bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=['Positive', 'Neutral', 'Negative'], 
                    y=probabilities, 
                    marker_color=['green', 'blue', 'red'],  # Colors for each category
                    text=probabilities,  # Show values on the bars
                    textposition='auto'
                )
            ])

            # Customize layout
            fig.update_layout(
                # title="Sentiment Analysis Results",
                xaxis_title="Sentiment Categories",
                yaxis_title="Probability",
                template="plotly_white"
            )

            # Show the figure

            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error("An error occurred: " + str(e))
