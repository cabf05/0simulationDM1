import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# =========================================================
# CONFIGURAÇÃO
# =========================================================
st.set_page_config("Insulin Pump Clinical Simulator", layout="wide")

DT = 5                 # minutos
STEPS_PER_DAY = 288    # 24h / 5min
TARGET_GLUCOSE = 110

# =========================================================
# MODELOS FISIOLÓGICOS
# =========================================================
class PhysiologyState:
    def __init__(self):
        self.glucose = 110.0
        self.insulin = np.zeros(4)   # compartimentos de insulina
        self.carbs = np.zeros(2)     # compartimentos de carboidrato

    def to_dict(self):
        return {
            "glucose": self.glucose,
            "insulin": self.insulin.tolist(),
            "carbs": self.carbs.tolist()
        }

    @staticmethod
    def from_dict(d):
        s = PhysiologyState()
        s.glucose = d["glucose"]
        s.insulin = np.array(d["insulin"])
        s.carbs = np.array(d["carbs"])
        return s


class PatientProfile:
    def __init__(self, bolus_delay, variability):
        self.bolus_delay = bolus_delay
        self.variability = variability
        self.isf = 40         # mg/dL por U
        self.carb_abs = [0.03, 0.01]  # rápida / lenta


class PumpSettings:
    def __init__(self, basal, ic):
        self.basal = basal   # U/h
        self.ic = ic         # g/U


# =========================================================
# MOTOR DE SIMULAÇÃO
# =========================================================
def step_simulation(state, patient, pump, pump_type):
    # Produção hepática basal
    hepatic = 0.8

    # Basal
    basal_u = pump.basal * DT / 60
    state.insulin[0] += basal_u

    # Automação simplificada
    if pump_type == "Suspensão automática" and state.glucose < 70:
        state.insulin[0] -= basal_u
    elif pump_type == "Híbrido (AID)":
        if state.glucose > 160:
            state.insulin[0] += 0.05
        if state.glucose < 80:
            state.insulin[0] -= 0.05

    # Dinâmica da insulina (compartimentos)
    k = [0.25, 0.20, 0.15, 0.10]
    for i in range(3, 0, -1):
        transfer = state.insulin[i-1] * k[i-1]
        state.insulin[i] += transfer
        state.insulin[i-1] -= transfer

    insulin_effect = state.insulin.sum() * patient.isf * DT / 240

    # Carboidratos
    carb_absorbed = 0
    for i in range(2):
        absorbed = state.carbs[i] * patient.carb_abs[i]
        carb_absorbed += absorbed
        state.carbs[i] -= absorbed

    # Variabilidade fisiológica
    noise = np.random.normal(0, patient.variability)

    # Atualização glicêmica
    state.glucose += carb_absorbed
    state.glucose -= insulin_effect
    state.glucose += hepatic
    state.glucose += noise

    state.glucose = max(40, state.glucose)


def simulate_consultation(state, patient, pump, pump_type, days):
    records = []

    meals = [
        (8 * 60, 50),
        (13 * 60, 70),
        (19 * 60, 60)
    ]

    for day in range(days):
        for step in range(STEPS_PER_DAY):
            t = step * DT

            for meal_time, carbs in meals:
                if t == meal_time:
                    state.carbs[0] += carbs * 0.7
                    state.carbs[1] += carbs * 0.3

                if t == meal_time + patient.bolus_delay:
                    bolus = carbs / pump.ic
                    state.insulin[0] += bolus

            step_simulation(state, patient, pump, pump_type)

            records.append({
                "day": day + 1,
                "minute": t,
                "glucose": state.glucose
            })

    return state, pd.DataFrame(records)


# =========================================================
# RESUMO CLÍNICO
# =========================================================
def clinical_summary(df):
    mean = df.glucose.mean()
    tir = ((df.glucose >= 70) & (df.glucose <= 180)).mean() * 100
    hypos = (df.glucose < 70).sum()
    hypers = (df.glucose > 180).sum()

    summary = []
    if mean > 160:
        summary.append("Hiperglicemia média elevada")
    if hypos > 5:
        summary.append("Hipoglicemias frequentes")
    if tir < 60:
        summary.append("Tempo em alvo reduzido")

    return {
        "mean": round(mean, 1),
        "tir": round(tir, 1),
        "hypos": int(hypos),
        "hypers": int(hypers),
        "notes": summary
    }


# =========================================================
# INTERFACE
# =========================================================
st.title("🩺 Insulin Pump Clinical Simulator")
st.markdown("**Treinamento em raciocínio clínico longitudinal**")

if "state" not in st.session_state:
    st.session_state.state = PhysiologyState()
    st.session_state.consult = 1
    st.session_state.history = []

st.sidebar.header("Configuração")

pump_type = st.sidebar.selectbox(
    "Tipo de bomba",
    ["Convencional", "Suspensão automática", "Híbrido (AID)"]
)

days = st.sidebar.slider("Duração da consulta (dias)", 7, 30, 14)

basal = st.sidebar.slider("Basal (U/h)", 0.5, 2.0, 1.0, 0.1)
ic = st.sidebar.slider("IC (g/U)", 5, 20, 10)

delay = st.sidebar.slider("Atraso de bolus (min)", 0, 30, 15)
variability = st.sidebar.slider("Variabilidade fisiológica", 0.0, 5.0, 1.0)

patient = PatientProfile(delay, variability)
pump = PumpSettings(basal, ic)

# =========================================================
# EXECUÇÃO DA CONSULTA
# =========================================================
if st.button("▶️ Rodar próxima consulta"):
    state, df = simulate_consultation(
        st.session_state.state, patient, pump, pump_type, days
    )

    summary = clinical_summary(df)
    st.session_state.state = state
    st.session_state.history.append((df, summary))
    st.session_state.consult += 1

# =========================================================
# VISUALIZAÇÃO
# =========================================================
for i, (df, summary) in enumerate(st.session_state.history):
    st.subheader(f"Consulta {i+1}")

    fig, ax = plt.subplots()
    ax.plot(df.glucose)
    ax.axhline(70, linestyle="--")
    ax.axhline(180, linestyle="--")
    st.pyplot(fig)

    col1, col2, col3 = st.columns(3)
    col1.metric("Glicemia média", summary["mean"])
    col2.metric("TIR (%)", summary["tir"])
    col3.metric("Hipoglicemias", summary["hypos"])

    if summary["notes"]:
        st.info("Resumo clínico: " + " • ".join(summary["notes"]))

# =========================================================
# EXPORT / IMPORT
# =========================================================
st.subheader("📦 Exportar / Importar Simulação")

if st.session_state.history:
    export = {
        "state": st.session_state.state.to_dict(),
        "consult": st.session_state.consult,
        "history": [
            {
                "summary": s,
                "data": d.to_dict()
            }
            for d, s in st.session_state.history
        ]
    }

    df_export = pd.DataFrame([export])
    st.download_button(
        "Exportar simulação (CSV)",
        df_export.to_csv(index=False),
        "simulation_snapshot.csv",
        "text/csv"
    )

uploaded = st.file_uploader("Importar simulação", type="csv")

if uploaded:
    raw = pd.read_csv(uploaded).iloc[0]
    st.session_state.state = PhysiologyState.from_dict(eval(raw["state"]))
    st.session_state.consult = raw["consult"]
    st.session_state.history = []
    st.success("Simulação restaurada com sucesso")
