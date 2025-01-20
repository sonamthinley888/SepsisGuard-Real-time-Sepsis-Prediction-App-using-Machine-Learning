import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder

# Web App Title
st.title('Exploratory Data Analysis')

st.markdown('The exploratory data analysis will seek to answer the following questions: \n'
            '1. What is the number of attributes and observations in the dataset?\n'
            '2. Are there any missing values in the dataset?\n'
            '3. How many of the patients are diagnosed with Sepsis?\n'
            '4. What is the correlation between the variables?\n'
            '5. What is the distribution of the variables?')

# Load Dataset
sepsis_data = pd.read_csv("sepsis_file.csv")
df = sepsis_data.copy()

# Data Fields
st.subheader('Data Fields')
st.markdown('''
| Name        | Attribute/Target | Description |
|-------------|-----------------|-------------|
| ID          | N/A             | Unique patient ID |
| PRG         | Attribute 1      | Plasma glucose |
| PL          | Attribute 2      | Blood Work Result-1 (mu U/ml) |
| PR          | Attribute 3      | Blood Pressure (mm Hg) |
| SK          | Attribute 4      | Blood Work Result-2 (mm) |
| TS          | Attribute 5      | Blood Work Result-3 (mu U/ml) |
| M11         | Attribute 6      | Body mass index (kg/m²) |
| BD2         | Attribute 7      | Blood Work Result-4 (mu U/ml) |
| Age         | Attribute 8      | Patient age (years) |
| Insurance   | N/A             | Valid insurance card holder |
| Sepsis      | Target          | Sepsis diagnosis (0: No, 1: Yes) |
''')

# Remove unnecessary columns
st.subheader('Removing Patient ID and Insurance')
df = df.drop(columns=['ID', 'Insurance'], errors='ignore')
st.write(df)

# Convert Target Variable to Integer
st.subheader('Converting Target Variable to Integer')
for col in df.select_dtypes(include=['object']).columns:
    df[col] = OrdinalEncoder().fit_transform(df[col].values.reshape(-1, 1))
st.write(df)

# Data Statistics
st.header('Data Statistics')
st.write(df.describe())

st.header('Dataset Overview')
st.subheader('Data Shape (Rows, Columns)')
st.write(df.shape)

# Check missing values
st.subheader('Q2. Are there any missing values in the dataset?')
st.write(df.isnull().sum())

# Drop rows with missing values
df = df.dropna()
st.text('Dropped rows with missing values.')
st.write(df.shape)

# Sepsis Distribution
st.subheader('Q3. How many patients are diagnosed with Sepsis?')
st.write(df['Sepsis'].value_counts())

# Bar chart for Sepsis
fig, ax = plt.subplots(figsize=(5, 4))
df['Sepsis'].value_counts().plot(kind='bar', ax=ax, color=['blue', 'red'])
ax.set_title("Sepsis Diagnosis", fontsize=13, weight='bold')
ax.set_xticklabels(["Negative", "Positive"], rotation=0)
st.pyplot(fig)

# Correlation Heatmap
st.subheader('Q4. What is the correlation between the variables?')
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
plt.title('Correlation Between Variables', fontsize=15)
st.pyplot(fig)

st.text('Highest correlation: Blood Work Result 1 & Sepsis')
st.text('Plasma Glucose and Age show some correlation')

# Distribution of Variables
st.subheader('Q5. What is the distribution of the variables?')
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
axes = axes.flatten()

for i, col in enumerate(df.columns):
    if col != 'Sepsis':
        sns.histplot(df[col], ax=axes[i], bins=30, kde=True)
        axes[i].set_title(f"Distribution of {col}")

plt.tight_layout()
st.pyplot(fig)
