import numpy as np
import pandas as pd
import pickle
import streamlit as st

st.title("Predicción de ataque al corazón")

age = st.number_input("Edad", min_value=0, max_value=100, value=30, step=1)
avg_glucose_level = st.number_input(
    "Nivel promedio de glucosa", min_value=0.0, max_value=300.0, value=100.0, step=0.1
)
smoking_status = st.selectbox(
    "Estado de fumador",
    ["'formerly smoked'", "'never smoked'", "'Unknown'", "'smokes'"],
)
hypertension = st.selectbox("Hipertensión", ["No", "Yes"])
heart_disease = st.selectbox("Enfermedad cardíaca", ["No", "Yes"])
ever_married = st.selectbox("¿Alguna vez se casó?", ["No", "Yes"])

if st.button("Predecir"):
    with open("modelo-SVM.pkl", "rb") as file:
        modelo, labelencoder, variables, escalador = pickle.load(file)

    datos = [
        [
            age,
            avg_glucose_level,
            smoking_status,
            hypertension,
            heart_disease,
            ever_married,
        ]
    ]
    data = pd.DataFrame(
        datos,
        columns=[
            "age",
            "avg_glucose_level",
            "smoking_status",
            "hypertension",
            "heart_disease",
            "ever_married",
        ],
    )

    data_preparada = data.copy()
    data_preparada = pd.get_dummies(
        data_preparada,
        columns=["smoking_status", "hypertension", "heart_disease", "ever_married"],
        drop_first=False,
        dtype=int,
    )
    data_preparada = data_preparada.reindex(columns=variables, fill_value=0)

    data_preparada[["age", "avg_glucose_level"]] = escalador.transform(
        data_preparada[["age", "avg_glucose_level"]]
    )

    prediccion = modelo.predict(data_preparada)
    label = labelencoder.inverse_transform(prediccion)[0]

    st.success(f"Clase predicha: {label}")
