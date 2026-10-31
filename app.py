import pickle

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Riesgo de enfermedad cardíaca", page_icon="🫀", layout="centered")

# 1. Cargar el modelo entrenado: [Pipeline (StandardScaler + LogisticRegression), variables, umbral]
with open("modelo-lr-clf.pkl", "rb") as f:
    modelo, variables, umbral = pickle.load(f)

st.title("Predicción de enfermedad cardíaca")
st.caption("Modelo de clasificación (regresión logística) entrenado con el Heart Failure Prediction Dataset (918 pacientes).")

# 2. Capturar los datos del paciente (rangos según el dataset)
st.subheader("Datos del paciente")
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Edad (años)", 28, 77, 54)
    sex = st.selectbox("Sexo", ["M", "F"], format_func=lambda v: {"M": "Masculino", "F": "Femenino"}[v])
    chest = st.selectbox("Tipo de dolor torácico", ["ASY", "ATA", "NAP", "TA"],
                         format_func=lambda v: {"ASY": "Asintomático (ASY)", "ATA": "Angina atípica (ATA)",
                                                "NAP": "Dolor no anginoso (NAP)", "TA": "Angina típica (TA)"}[v])
    resting_bp = st.slider("Presión arterial en reposo (mm Hg)", 80, 200, 130)
    fasting_bs = st.selectbox("Glicemia en ayunas > 120 mg/dl", [0, 1], format_func=lambda v: "Sí" if v == 1 else "No")
with col2:
    resting_ecg = st.selectbox("Electrocardiograma en reposo", ["Normal", "ST", "LVH"],
                               format_func=lambda v: {"Normal": "Normal", "ST": "Anomalía ST-T",
                                                      "LVH": "Hipertrofia ventricular izquierda"}[v])
    max_hr = st.slider("Frecuencia cardíaca máxima (lpm)", 60, 202, 138)
    exercise_angina = st.selectbox("Angina inducida por ejercicio", ["N", "Y"], format_func=lambda v: "Sí" if v == "Y" else "No")
    oldpeak = st.slider("Oldpeak (depresión del ST, mm)", -2.6, 6.2, 0.6, step=0.1)
    st_slope = st.selectbox("Pendiente del segmento ST", ["Up", "Flat", "Down"],
                            format_func=lambda v: {"Up": "Ascendente", "Flat": "Plana", "Down": "Descendente"}[v])

paciente = pd.DataFrame([{
    "Age": age, "Sex": sex, "ChestPainType": chest, "RestingBP": resting_bp, "FastingBS": fasting_bs,
    "RestingECG": resting_ecg, "MaxHR": max_hr, "ExerciseAngina": exercise_angina,
    "Oldpeak": oldpeak, "ST_Slope": st_slope,
}])

# 3. Crear las variables dummy y ordenarlas como en el entrenamiento
X = pd.get_dummies(paciente, columns=["ChestPainType", "RestingECG", "ST_Slope", "Sex", "ExerciseAngina"], dtype=int)
X = X.reindex(columns=variables, fill_value=0)

# 4. Predecir (el Pipeline solo transforma; no se vuelve a entrenar)
if st.button("Predecir", type="primary"):
    prob = modelo.predict_proba(X)[0, 1]
    st.subheader("Resultado")
    st.metric("Probabilidad de enfermedad cardíaca", f"{prob * 100:.1f} %")
    if prob >= umbral:
        st.error(f"Riesgo ALTO: la probabilidad supera el umbral de decisión ({umbral:.0%}). "
                 "Se recomienda remitir al paciente a estudios confirmatorios.")
    else:
        st.success(f"Riesgo BAJO: la probabilidad está por debajo del umbral de decisión ({umbral:.0%}).")
    st.info("Desempeño del modelo en validación cruzada: exactitud 86.6 %, recall 90.6 %, precisión 86.0 %, AUC 0.92. "
            "Herramienta de apoyo con fines académicos: no reemplaza el criterio médico.")
