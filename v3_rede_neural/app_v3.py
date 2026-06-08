import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json

st.set_page_config(page_title="Burnout Radar v3", page_icon="🧠", layout="wide")

st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  h1, h2, h3 { font-family: 'Space Mono', monospace; }
  .result-card { border-radius: 16px; padding: 2rem; margin-top: 1.5rem; text-align: center; }
  .card-high   { background: linear-gradient(135deg, #ff2d55, #8b0000); color: white; }
  .card-medium { background: linear-gradient(135deg, #ff9f0a, #7d4800); color: white; }
  .card-low    { background: linear-gradient(135deg, #34c759, #004d1f); color: white; }
  .result-label { font-size: 2.5rem; font-weight: 700; margin: 0.5rem 0; font-family: 'Space Mono', monospace; }
  .stButton > button { width: 100%; background: linear-gradient(135deg, #6c63ff, #3b35cc); color: white; font-weight: 700; border: none; border-radius: 10px; padding: 0.75rem; }
  .badge { display: inline-block; background: #6c63ff22; border: 1px solid #6c63ff; color: #a89fff; border-radius: 8px; padding: 2px 10px; font-size: 0.8rem; font-family: 'Space Mono', monospace; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Definição da rede (deve ser idêntica à usada no treino) ──────────────────
class RedeNeuralBurnout(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super(RedeNeuralBurnout, self).__init__()
        self.rede = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.rede(x)


# ── Carregamento do modelo ────────────────────────────────────────────────────
@st.cache_resource
def carregar_modelo():
    with open("model_metadata_v3.json") as f:
        meta = json.load(f)

    checkpoint = torch.load("modelo_burnout_v3.pth", map_location="cpu")
    modelo = RedeNeuralBurnout(checkpoint["input_dim"], checkpoint["num_classes"])
    modelo.load_state_dict(checkpoint["model_state_dict"])
    modelo.eval()

    preprocessor = joblib.load("preprocessor_v3.pkl")
    le = joblib.load("label_encoder_v3.pkl")
    return modelo, preprocessor, le, meta


try:
    modelo, preprocessor, le, meta = carregar_modelo()
    modelo_ok = True
except Exception as e:
    modelo_ok = False
    st.error(f"Modelo não encontrado: {e}")
    st.info(
        "Rode o notebook_v3.ipynb no Kaggle e baixe: modelo_burnout_v3.pth · preprocessor_v3.pkl · label_encoder_v3.pkl · model_metadata_v3.json"
    )
    st.stop()


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧠 Burnout Radar — v3")
st.markdown(
    "##### Rede Neural (PyTorch) · Loop de Feedback · <span class='badge'>Deep Learning</span>",
    unsafe_allow_html=True,
)
st.markdown("---")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Dados do Aluno")

    st.markdown("**📚 Estudo & IA**")
    study_hours = st.slider("Horas de Estudo Tradicional / semana", 0, 60, 15)
    ai_hours = st.slider("Horas de Uso de IA Generativa / semana", 0, 40, 5)
    ai_dependency = st.slider("Dependência da IA (1–5)", 1, 5, 3)

    st.markdown("**📝 Desempenho**")
    pre_gpa = st.slider("GPA pré-semestre (0.0–4.0)", 0.0, 4.0, 3.0, step=0.1)
    post_gpa = st.slider("GPA pós-semestre (0.0–4.0)", 0.0, 4.0, 3.0, step=0.1)
    anxiety = st.slider("Ansiedade nos Exames (1–10)", 1, 10, 5)
    skill_retention = st.slider("Retenção de Habilidades (0–100)", 0, 100, 75)
    tool_diversity = st.slider("Diversidade de Ferramentas (1–5)", 1, 5, 2)

    st.markdown("**🎓 Perfil**")
    major = st.selectbox(
        "Área de Estudo", ["STEM", "Business", "Humanities", "Medical", "Arts"]
    )
    year = st.selectbox(
        "Ano na Faculdade", ["Freshman", "Sophomore", "Junior", "Senior"]
    )
    use_case = st.selectbox(
        "Uso Principal da IA",
        [
            "Copywriting/Drafting",
            "Summarizing_Reading",
            "Debugging/Troubleshooting",
            "Ideation",
            "Research",
        ],
    )
    prompt_skill = st.selectbox(
        "Habilidade em Prompts", ["Beginner", "Intermediate", "Advanced"]
    )
    policy = st.selectbox(
        "Política Institucional",
        ["Allowed_With_Citation", "Strict_Ban", "Actively_Encouraged"],
    )

    prever = st.button("🔮 Prever Risco de Burnout")


# ── Área principal ────────────────────────────────────────────────────────────
col_esq, col_dir = st.columns([3, 2], gap="large")

with col_esq:
    st.markdown("### Painel do Aluno")

    ai_ratio = ai_hours / (study_hours + 1)
    dep_ratio = ai_dependency * ai_ratio

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Razão IA/Estudo", f"{ai_ratio:.2f}")
    m2.metric("Dep × Razão", f"{dep_ratio:.2f}")
    m3.metric("GPA Pré", f"{pre_gpa:.1f}")
    m4.metric("GPA Pós", f"{post_gpa:.1f}")

    if prever:
        row = {
            "Traditional_Study_Hours": study_hours,
            "Weekly_GenAI_Hours": ai_hours,
            "AI_Ratio": ai_ratio,
            "AI_Dependency_x_Ratio": dep_ratio,
            "Perceived_AI_Dependency": ai_dependency,
            "Pre_Semester_GPA": pre_gpa,
            "Post_Semester_GPA": post_gpa,
            "Anxiety_Level_During_Exams": anxiety,
            "Skill_Retention_Score": skill_retention,
            "Tool_Diversity": tool_diversity,
            "Major_Category": major,
            "Year_of_Study": year,
            "Primary_Use_Case": use_case,
            "Prompt_Engineering_Skill": prompt_skill,
            "Institutional_Policy": policy,
        }

        todas_cols = meta["colunas_numericas"] + meta["colunas_categoricas"]
        entrada_df = pd.DataFrame([{k: v for k, v in row.items() if k in todas_cols}])

        # Pré-processa e converte para tensor
        X_proc = preprocessor.transform(entrada_df)
        X_tensor = torch.FloatTensor(X_proc)

        # Predição
        with torch.no_grad():
            saida = modelo(X_tensor)
            proba = torch.softmax(saida, dim=1).numpy()[0]
            classe_idx = int(torch.argmax(saida, dim=1).item())

        previsao = le.classes_[classe_idx]

        card_class = {"High": "card-high", "Medium": "card-medium", "Low": "card-low"}
        emoji = {"High": "🚨", "Medium": "⚠️", "Low": "✅"}
        label_pt = {
            "High": "Alto Risco",
            "Medium": "Risco Moderado",
            "Low": "Baixo Risco",
        }

        st.markdown(
            f"""
        <div class="result-card {card_class[previsao]}">
            <div>NÍVEL DE BURNOUT PREVISTO</div>
            <div class="result-label">{emoji[previsao]} {label_pt[previsao]}</div>
            <div>Classe: <strong>{previsao}</strong></div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("**Confiança da Rede Neural por Classe**")
        for cls, prob in zip(le.classes_, proba):
            st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")

        st.markdown("---")
        st.markdown("#### 💡 Recomendações")
        if previsao == "High":
            st.error("**Atenção imediata necessária!**")
            for r in [
                "🛑 Limite o uso de IA para tarefas repetitivas",
                "😴 Priorize sono — menos de 7h/noite amplifica o burnout",
                "🤝 Aumente interações sociais presenciais",
                "📅 Considere conversar com um orientador acadêmico",
            ]:
                st.markdown(f"- {r}")
        elif previsao == "Medium":
            st.warning("**Risco moderado — ajuste os hábitos agora**")
            for r in [
                "⚖️ Equilibre horas de IA e estudo tradicional",
                "📝 Mantenha taxa de conclusão acima de 80%",
                "🧘 Pratique técnicas de relaxamento 15min/dia",
            ]:
                st.markdown(f"- {r}")
        else:
            st.success("**Excelente! Continue assim**")
            for r in [
                "🌟 Seus hábitos estão equilibrados",
                "📈 Monitore seu estado mensalmente",
                "🤖 Você usa IA de forma saudável!",
            ]:
                st.markdown(f"- {r}")
    else:
        st.info("👈 Preencha os dados na barra lateral e clique em Prever")

with col_dir:
    st.markdown("### Sobre o Modelo v3")
    st.markdown("""
| Item | v3 |
|------|----|
| Algoritmo | Rede Neural (PyTorch) |
| Camadas | Input → 128 → 64 → 3 |
| Regularização | BatchNorm + Dropout |
| Features | 15 |
| Treino | 30 épocas iniciais |
| Feedback | 3 rodadas incrementais |
| Acurácia | ~70% |
""")

    st.markdown("---")
    st.markdown("### 🗺️ Roadmap")
    st.markdown("""
- [x] v1 · Random Forest (4 features)
- [x] v2 · Gradient Boosting (15 features)
- [x] v3 · Rede Neural + Loop de Feedback
- [ ] v4 · Deploy em nuvem (Hugging Face)
""")
    st.markdown("---")
    st.caption("Dataset: AI Impact on Students · Kaggle")
    st.caption("Curso Intensivo ML · Prof. Nirmal Gaud")
    st.caption("Desenvolvido por Aline Lima 🇧🇷")
