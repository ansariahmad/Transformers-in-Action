import numpy as np
import streamlit as st
import plotly.graph_objects as go
from transformers import pipeline

st.set_page_config(
    page_title="Fill Mask",
    page_icon="🚀")

# App Title
st.title("Zero-Shot Text Classification")

# Initialize the zero-shot classification pipeline
zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Colors
colors = ['rgba(24, 203, 162, 1)', 'rgba(34, 180, 20, 1)', 'rgba(231, 110, 212, 1)', 'rgba(191, 206, 164, 1)', 'rgba(100, 233, 42, 1)', 
    'rgba(185, 222, 92, 1)', 'rgba(27, 157, 138, 1)', 'rgba(212, 207, 155, 1)', 'rgba(172, 202, 164, 1)', 'rgba(47, 65, 177, 1)', 
    'rgba(26, 44, 233, 1)', 'rgba(65, 242, 9, 1)', 'rgba(171, 50, 253, 1)', 'rgba(125, 201, 227, 1)', 'rgba(135, 196, 15, 1)', 
    'rgba(114, 106, 242, 1)', 'rgba(176, 50, 34, 1)', 'rgba(100, 159, 247, 1)', 'rgba(246, 103, 72, 1)', 'rgba(180, 180, 5, 1)', 
    'rgba(64, 29, 164, 1)', 'rgba(65, 192, 5, 1)', 'rgba(149, 97, 155, 1)', 'rgba(210, 2, 107, 1)', 'rgba(70, 203, 162, 1)', 
    'rgba(68, 74, 64, 1)', 'rgba(164, 42, 173, 1)', 'rgba(220, 37, 239, 1)', 'rgba(76, 89, 84, 1)', 'rgba(29, 190, 84, 1)', 
    'rgba(180, 35, 240, 1)', 'rgba(222, 72, 217, 1)', 'rgba(203, 80, 243, 1)', 'rgba(121, 164, 68, 1)', 'rgba(107, 218, 79, 1)', 
    'rgba(152, 225, 65, 1)', 'rgba(57, 170, 43, 1)', 'rgba(77, 131, 61, 1)', 'rgba(145, 101, 161, 1)', 'rgba(115, 77, 3, 1)', 
    'rgba(29, 159, 63, 1)', 'rgba(71, 105, 200, 1)', 'rgba(98, 78, 55, 1)', 'rgba(242, 159, 60, 1)', 'rgba(175, 67, 54, 1)', 
    'rgba(120, 246, 81, 1)', 'rgba(216, 132, 219, 1)', 'rgba(82, 77, 251, 1)', 'rgba(213, 29, 120, 1)', 'rgba(252, 90, 31, 1)', 
    'rgba(194, 181, 168, 1)', 'rgba(246, 60, 189, 1)', 'rgba(22, 50, 26, 1)', 'rgba(54, 11, 134, 1)', 'rgba(27, 103, 59, 1)', 
    'rgba(234, 96, 187, 1)', 'rgba(167, 157, 215, 1)', 'rgba(104, 1, 252, 1)', 'rgba(76, 121, 131, 1)', 'rgba(65, 250, 218, 1)', 
    'rgba(219, 59, 127, 1)', 'rgba(18, 242, 194, 1)', 'rgba(14, 132, 131, 1)', 'rgba(82, 68, 61, 1)', 'rgba(109, 229, 43, 1)', 
    'rgba(202, 96, 66, 1)', 'rgba(216, 112, 64, 1)', 'rgba(101, 215, 114, 1)', 'rgba(85, 234, 109, 1)', 'rgba(17, 43, 113, 1)', 
    'rgba(104, 132, 5, 1)', 'rgba(23, 177, 214, 1)', 'rgba(112, 131, 160, 1)', 'rgba(142, 43, 188, 1)', 'rgba(189, 61, 176, 1)', 
    'rgba(196, 198, 61, 1)', 'rgba(253, 176, 165, 1)', 'rgba(113, 143, 126, 1)', 'rgba(122, 156, 220, 1)', 'rgba(221, 11, 29, 1)', 
    'rgba(233, 200, 5, 1)', 'rgba(232, 176, 217, 1)', 'rgba(199, 6, 130, 1)', 'rgba(140, 118, 154, 1)', 'rgba(177, 46, 36, 1)', 
    'rgba(244, 81, 66, 1)', 'rgba(94, 99, 24, 1)', 'rgba(159, 90, 50, 1)', 'rgba(67, 144, 236, 1)', 'rgba(78, 202, 143, 1)', 
    'rgba(13, 116, 114, 1)', 'rgba(139, 194, 124, 1)', 'rgba(174, 63, 214, 1)', 'rgba(84, 114, 130, 1)', 'rgba(143, 208, 199, 1)', 
    'rgba(27, 60, 225, 1)', 'rgba(69, 228, 28, 1)', 'rgba(167, 157, 10, 1)', 'rgba(61, 185, 55, 1)', 'rgba(143, 52, 233, 1)']

colors = np.array(colors)

# Input Section
st.write("Enter a sentence or text to classify and provide possible labels.")

user_input = st.text_input("Input your text:", "Streamlit is an amazing tool for building web apps.")
labels_input = st.text_input("Enter possible labels (comma-separated):", "technology, finance, health")

# Process and Display Results
if st.button("Classify Text"):
    labels = [label.strip().title() for label in labels_input.split(",") if label.strip()]
    if not user_input or not labels:
        st.error("Please provide both text and at least one label.")
    else:
        try:
            st.write("## Classification Results:")
            probabilities = []
            result = zero_shot(user_input, labels)

            for label, score in zip(result['labels'], result['scores']):
                probabilities.append(round(score, 2))
            
            fig = go.Figure(data=[
            go.Bar(
                x=labels, 
                y=probabilities, 
                marker_color=np.random.choice(colors, len(labels)).tolist(),  # Colors for each category
                text=probabilities,  # Show values on the bars
                textposition='auto'
            )
        ])

        # Customize layout
            fig.update_layout(
                # title="Sentiment Analysis Results",
                xaxis_title="Label",
                yaxis_title="Probability",
                template="seaborn",
            )

            # Show the figure

            st.plotly_chart(fig, use_container_width=True, theme=None)

        except Exception as e:
                st.error(f"An error occurred: {e}")
            
