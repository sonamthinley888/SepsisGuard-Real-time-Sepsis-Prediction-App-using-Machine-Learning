import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder

# Load Data
sepsis_data = pd.read_csv("sepsis_file.csv")
df = sepsis_data.copy()

# Attribute to be predicted
predict = "Sepsis"

# Streamlit title
st.title('Predictive Data Analysis')

# Data Preprocessing
st.subheader('Pre-Processing Data')

# Drop columns that aren't useful for prediction (if necessary)
df = df[["PRG", "PL", "PR", "SK", "TS", "M11", "BD2", "Age", "Sepsis"]]

# Convert Target Variable to Integer
st.subheader('Converting Target Variable to Integer')
for col in df.select_dtypes(include=['object']).columns:
    df[col] = OrdinalEncoder().fit_transform(df[col].values.reshape(-1, 1))
st.write(df)

# Separate features and target
class_label = df['Sepsis']
df = df.drop(['Sepsis'], axis=1)

# Normalize features (exclude Sepsis)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Reattach the target column after scaling
df_scaled['Sepsis'] = class_label



# Visualize correlation matrix
st.subheader('Correlation Matrix')
correlation_matrix = df_scaled.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
st.pyplot()

# Handle class imbalance using SMOTE (if needed)
smote = SMOTE(random_state=42)
X = df_scaled.drop('Sepsis', axis=1)
y = df_scaled['Sepsis']
X_res, y_res = smote.fit_resample(X, y)

# Split data into train and test sets
st.subheader('Train-Test Split')
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.20, random_state=42)

# Display the size of train and test sets
st.write(f"Train set size: {X_train.shape}, Test set size: {X_test.shape}")

# Models to evaluate
models = [
    ('SVM', SVC()),
    ('Random Forest', RandomForestClassifier()),
    ('Gradient Boosting', GradientBoostingClassifier())
]

# Hyperparameter tuning using GridSearchCV
st.subheader('Hyperparameter Tuning using GridSearchCV')

param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [10, 20], 'min_samples_split': [2, 5]}
param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
param_grid_gbm = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]}

# RandomForest
grid_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=3)
grid_rf.fit(X_train, y_train)

# SVM
grid_svm = GridSearchCV(SVC(), param_grid_svm, cv=3)
grid_svm.fit(X_train, y_train)

# GradientBoosting
grid_gbm = GridSearchCV(GradientBoostingClassifier(), param_grid_gbm, cv=3)
grid_gbm.fit(X_train, y_train)

# Evaluate the best models
best_model_rf = grid_rf.best_estimator_
best_model_svm = grid_svm.best_estimator_
best_model_gbm = grid_gbm.best_estimator_

# Test models on the test set
st.subheader('Model Evaluation on Test Set')
y_pred_rf = best_model_rf.predict(X_test)
y_pred_svm = best_model_svm.predict(X_test)
y_pred_gbm = best_model_gbm.predict(X_test)

# Calculate Accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_gbm = accuracy_score(y_test, y_pred_gbm)

st.write(f"Random Forest Accuracy: {accuracy_rf}")
st.write(f"SVM Accuracy: {accuracy_svm}")
st.write(f"Gradient Boosting Accuracy: {accuracy_gbm}")

# Display classification report and confusion matrix for each model
models = ['Random Forest', 'SVM', 'Gradient Boosting']
y_preds = [y_pred_rf, y_pred_svm, y_pred_gbm]

for model, y_pred in zip(models, y_preds):
    st.subheader(f'{model} - Classification Report')
    report = classification_report(y_test, y_pred)
    st.text(report)
    
    st.subheader(f'{model} - Confusion Matrix')
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    st.pyplot()

    # ROC-AUC score
    st.write(f'{model} - ROC-AUC Score: {roc_auc_score(y_test, y_pred)}')

# Boxplot to compare models
st.subheader('Model Comparison')
accuracy_scores = [accuracy_rf, accuracy_svm, accuracy_gbm]
fig = plt.figure()
plt.boxplot(accuracy_scores)
plt.xticks([1, 2, 3], models)
plt.title('Model Comparison')
st.pyplot(fig)
