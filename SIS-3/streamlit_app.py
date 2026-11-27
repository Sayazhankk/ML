import os

import requests
import streamlit as st

# Адрес FastAPI (по умолчанию - локальный)
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

st.title("Iris Flower Classifier")
st.write("Введите размеры цветка и нажмите кнопку.")

sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2)

if st.button("Predict"):
    data = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }
    try:
        response = requests.post(API_URL, json=data)
        result = response.json()
        st.success(f"Class: {result['class_name']}")
        st.write(f"Probability: {result['probability']}")
    except Exception:
        st.error("Не удалось подключиться к API. Запустите: uvicorn main:app --reload")
