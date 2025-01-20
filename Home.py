import streamlit as st
# Please run this in terminal : streamlit run Home.py
# The components of this Home page is in the directory: pages



st.title('Title: Sepsis Prediction using Sepsis Database')
st.markdown('Author: Mr. Sonam Thinley')
st.markdown('This report describes a Python Capstone Project '
        'focused on sepsis detection using a dataset from Kaggle. '
        'Sepsis is a serious and life-threatening reaction to an infection '
        'and is a leading cause of death in hospitals. '
        'Detecting sepsis early is critical, '
        'and a sepsis prediction tool could help by analyzing vital signs and clinical data. '
        'The report discusses the challenges of identifying sepsis and'
        ' the potential benefits of a prediction tool. '
        'The report also presents a prototype software platform '
        'that uses Python tools and a data-driven approach '
        'to perform exploratory data analysis and predictive analytics. '
        'The platform includes a desktop Tkinter application '
        'and an online web-based Streamlit application.')

st.subheader('Methodology')
st.markdown('The Methodology used to develop the software platform '
        'consists of three main phases, as described below:')

st.markdown('•	Creating decision support algorithms by utilizing exploratory data analysis'
        ' and predictive analytics. This step involves identifying '
        'the best algorithm that can solve a real-world problem.')
st.markdown('•	Developing a desktop Tkinter software tool using the best-performing algorithm.')
st.markdown('•	Deploying the tool as a web or cloud-enabled platform.')


