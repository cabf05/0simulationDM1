import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# ---------------------------
# CONFIGURAÇÃO GERAL
# ---------------------------
st.set_page_config(page_title="Insulin Pump Clinical Simulator", layout="wide")

TIME_STEP_MIN = 5
DAY_MINUTES = 24 * 60
STEPS_PER_DAY = DAY_MINUTES // TIME_STEP_MIN

# ---------------------------
# MODELOS CLÍNICOS
# ---------------------------
class PatientProfile:
    def __init__(self, bolus_delay_min=0):
        self.age = 35
        self.weight = 75
        self.isf = 40          # mg/dL por U
        self.carb_abs_min = 90
        self.insulin_action_min = 240
        self.target_glucose = 110
        self.bolus_delay_min = bolus_delay_min


class PumpSettings:
    def __init__(self, basal=1.0, ic=10):
        self.basal = basal     # U/h
        self.ic = ic           # g/U


# ---------------------------
# SIMULAÇÃO
# ---------------------------
def simulate(patient, pump, days, pump_type):
    glucose = 110
    IOB = 0
    COB = 0

    history = []

    meals = [
        {"time": 8 * 60, "carbs": 50},
        {"time": 13 * 60, "carbs": 70},
        {"time": 19 * 60, "carbs": 60},
    ]

    for day in range(days):
        for step in range(STEPS_PER_DAY):
            time_min = step * TIME_STEP_MIN

            # Basal
            basal_u = pump.basal * (TIME_STEP_MIN / 60)
            IOB += basal_u

            # Automação conforme tipo de bomba
            if pump_type == "Suspensão automática" and glucose < 70:
                IOB -= basal_u  # suspende basal
            elif pump_type == "Híbrido (AID)":
                if glucose > 150:
                    IOB += 0.05
                if glucose < 80:
                    IOB -= 0.05

            # Refeições
            for meal in meals:
                if time_min == meal["time"]:
                    COB += meal["carbs"]

                bolus_time = meal["time"] + patient.bolus_delay_min
                if time_min == bolus_time:
                    bolus = meal["carbs"] / pump.ic
                    IOB += bolus

            # Absorção de carboidrato
            absorbed = COB * (TIME_STEP_MIN / patient.carb_abs_min)
            COB -= absorbed
            glucose += absorbed

            # Ação da insulina
            insulin_effect = IOB * (TIME_STEP_MIN / patient.insulin_action_min) * patient.isf
            glucose -= insulin_effect
            IOB -= IOB * (TIME_STEP_MIN / patient.insulin_action_min)

            history.append({
                "day": day + 1,
                "minute": time_min,
                "glucose": glucose,
                "IOB": IOB
            })

    return pd.DataFrame(history)


# ---------------------------
# INTERFACE
# ---------------------------
st.title("🩺 Insulin Pump Clinical Simulator")
st.markdown("**Foco: raciocínio clínico longitudinal — não operação do dispositivo**")

# Sidebar — parâmetros
st.sidebar.header("Configuração da Simulação")

pump_type = st.sidebar.selectbox(
    "Tipo de bomba",
    ["Convencional", "Suspensão automática", "Híbrido (AID)"]
)

days_per_consult = st.sidebar.slider("Duração de cada consulta (dias)", 7, 28, 14)

st.sidebar.subheader("Parâmetros Clínicos (editáveis)")
basal = st.sidebar.slider("Basal (U/h)", 0.5, 2.0, 1.0, 0.1)
ic = st.sidebar.slider("IC (g/U)", 5, 20, 10)

# Pacientes
st.sidebar.subheader("Perfis de Paciente")
delay = st.sidebar.slider("Atraso de bolus (min)", 0, 30, 15)

patient_A = PatientProfile(bolus_delay_min=0)
patient_B = PatientProfile(bolus_delay_min=delay)

pump = PumpSettings(basal=basal, ic=ic)

# ---------------------------
# SIMULAR
# ---------------------------
if st.button("▶️ Iniciar Simulação"):
    with st.spinner("Simulando..."):
        df_A = simulate(patient_A, pump, days_per_consult, pump_type)
        df_B = simulate(patient_B, pump, days_per_consult, pump_type)

    st.success("Simulação concluída")

    col1, col2 = st.columns(2)

    for df, title, col in [
        (df_A, "Paciente sem atraso", col1),
        (df_B, "Paciente com atraso", col2)
    ]:
        with col:
            st.subheader(title)

            fig, ax = plt.subplots()
            ax.plot(df["glucose"])
            ax.axhline(70, linestyle="--")
            ax.axhline(180, linestyle="--")
            ax.set_ylabel("Glicemia (mg/dL)")
            ax.set_xlabel("Tempo (passos de 5 min)")
            st.pyplot(fig)

            st.metric("Média glicêmica", round(df["glucose"].mean(), 1))
            tir = ((df["glucose"] >= 70) & (df["glucose"] <= 180)).mean() * 100
            st.metric("TIR (%)", round(tir, 1))

    # Exportação
    st.subheader("📤 Exportar / 📥 Importar Simulação")

    csv = df_B.to_csv(index=False)
    st.download_button(
        label="Exportar simulação (CSV)",
        data=csv,
        file_name="simulacao.csv",
        mime="text/csv"
    )

# ---------------------------
# IMPORTAÇÃO
# ---------------------------
uploaded = st.file_uploader("Importar simulação (CSV)", type="csv")

if uploaded:
    df_import = pd.read_csv(uploaded)
    st.subheader("Simulação Importada")

    fig, ax = plt.subplots()
    ax.plot(df_import["glucose"])
    ax.axhline(70, linestyle="--")
    ax.axhline(180, linestyle="--")
    st.pyplot(fig)

    st.write(df_import.head())
