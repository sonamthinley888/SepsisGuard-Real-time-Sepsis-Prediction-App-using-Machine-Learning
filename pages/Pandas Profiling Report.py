import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
from streamlit.components.v1 import html

# Data file import
sepsis_data = pd.read_csv("sepsis_file.csv")
df = sepsis_data

# Removing Patient ID and Insurance
df = df[["PRG", "PL", "PR", "SK", "TS", "M11", "BD2", "Age", "Sepsis"]]

# Convert target variable to integer
from sklearn.preprocessing import OrdinalEncoder

for col in df:
    if df[col].dtype == 'object':
        df[col] = OrdinalEncoder().fit_transform(df[col].values.reshape(-1, 1))

# Pandas Profiling Report
pr = ProfileReport(df, explorative=True)

st.header('**Input DataFrame**')
st.write(df)
st.write('---')

st.header('**Pandas Profiling Report**')
st.subheader('Shows a detailed information about the dataset')

# Display Pandas Profiling report
html_content = pr.to_html()
st.components.v1.html(html_content, height=800, scrolling=True)
