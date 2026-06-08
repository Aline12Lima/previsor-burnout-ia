import streamlit as st
import pandas as pd
import joblib

modelo = joblib.load("modelo_burnout_v1.pkl")

st.title("🧠 Previsor de Burnout Estudantil — v1")
st.write("Versão 1 · Random Forest · 4 features")
st.write(
    "Modelo treinado com horas de estudo tradicional, uso de IA e dependência percebida."
)

st.sidebar.header("Dados do Aluno")
study_hours = st.sidebar.slider("Horas de Estudo Tradicional (semana)", 0, 50, 10)
ai_hours = st.sidebar.slider("Horas de Uso de IA Gen (semana)", 0, 50, 5)
ai_dependency = st.sidebar.slider("Nível de Dependência da IA (1 a 5)", 1, 5, 3)

ai_ratio = ai_hours / (study_hours + 1)

if st.button("🔮 Prever Risco de Burnout"):
    dados = pd.DataFrame(
        [[study_hours, ai_hours, ai_ratio, ai_dependency]],
        columns=[
            "Traditional_Study_Hours",
            "Weekly_GenAI_Hours",
            "AI_Ratio",
            "Perceived_AI_Dependency",
        ],
    )
    previsao = modelo.predict(dados)[0]

    if previsao == "High":
        st.error(f"🚨 Risco Previsto: {previsao} — Alto Risco de Esgotamento!")
        st.write(
            "Recomendação: Diminuir a dependência da IA e balancear as horas de estudo."
        )
    elif previsao == "Medium":
        st.warning(f"⚠️ Risco Previsto: {previsao} — Risco Moderado")
        st.write(
            "Recomendação: Monitore seus hábitos e tente equilibrar melhor o uso da IA."
        )
    else:
        st.success(f"✅ Risco Previsto: {previsao} — Baixo Risco")
        st.write("Continue assim! Seus hábitos estão equilibrados.")

st.markdown("---")
st.caption(
    "Dataset: AI Impact on Students · Kaggle | Curso Intensivo ML · Prof. Nirmal Gaud"
)
