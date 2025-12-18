# =========================================================
# INSULIN PUMP CLINICAL TRAINER – STREAMLIT APP
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURAÇÃO
# =========================================================
st.set_page_config("Insulin Pump Clinical Trainer", layout="wide")

DT = 5
STEPS_PER_DAY = 288

# =========================================================
# MODELOS FISIOLÓGICOS
# =========================================================
class PhysiologyState:
    def __init__(self):
        self.glucose = 110.0
        self.insulin = np.zeros(4)
        self.carbs = np.zeros(2)


class PatientProfile:
    def __init__(self, bolus_delay, variability):
        self.bolus_delay = bolus_delay
        self.variability = variability
        self.isf = 40
        self.carb_abs = [0.03, 0.01]


class PumpSettings:
    def __init__(self, basal, ic):
        self.basal = basal
        self.ic = ic


# =========================================================
# MOTOR DE SIMULAÇÃO
# =========================================================
def step_simulation(state, patient, pump, pump_type):
    hepatic = 0.8

    basal_u = pump.basal * DT / 60
    state.insulin[0] += basal_u

    if pump_type == "Suspensão automática" and state.glucose < 70:
        state.insulin[0] -= basal_u

    if pump_type == "Híbrido (AID)":
        if state.glucose > 160:
            state.insulin[0] += 0.05
        if state.glucose < 80:
            state.insulin[0] -= 0.05

    k = [0.25, 0.20, 0.15, 0.10]
    for i in range(3, 0, -1):
        transfer = state.insulin[i-1] * k[i-1]
        state.insulin[i] += transfer
        state.insulin[i-1] -= transfer

    insulin_effect = state.insulin.sum() * patient.isf * DT / 240

    carb_absorbed = 0
    for i in range(2):
        absorbed = state.carbs[i] * patient.carb_abs[i]
        carb_absorbed += absorbed
        state.carbs[i] -= absorbed

    noise = np.random.normal(0, patient.variability)

    state.glucose += carb_absorbed
    state.glucose -= insulin_effect
    state.glucose += hepatic
    state.glucose += noise

    state.glucose = max(40, state.glucose)


def simulate_consultation(state, patient, pump, pump_type, days):
    records = []

    meals = [(480, 50), (780, 70), (1140, 60)]

    for day in range(days):
        for step in range(STEPS_PER_DAY):
            t = step * DT

            for meal_time, carbs in meals:
                if t == meal_time:
                    state.carbs[0] += carbs * 0.7
                    state.carbs[1] += carbs * 0.3

                if t == meal_time + patient.bolus_delay:
                    state.insulin[0] += carbs / pump.ic

            step_simulation(state, patient, pump, pump_type)

            records.append({
                "minute": t,
                "glucose": state.glucose
            })

    return state, pd.DataFrame(records)


# =========================================================
# MÉTRICAS CLÍNICAS
# =========================================================
def clinical_metrics(df):
    mean = df.glucose.mean()
    cv = df.glucose.std() / mean * 100

    return {
        "mean": round(mean, 1),
        "tir": round(((df.glucose >= 70) & (df.glucose <= 180)).mean() * 100, 1),
        "tbr": round((df.glucose < 70).mean() * 100, 1),
        "tar": round((df.glucose > 180).mean() * 100, 1),
        "cv": round(cv, 1)
    }


# =========================================================
# AGP SIMPLIFICADO — CORRIGIDO
# =========================================================
def plot_agp(df):
    df = df.copy()
    df["hour"] = (df.minute // 60).astype(int)

    agp = (
        df.groupby("hour")["glucose"]
        .agg(
            median=np.median,
            p25=lambda x: np.percentile(x, 25),
            p75=lambda x: np.percentile(x, 75),
        )
        .reset_index()
    )

    fig, ax = plt.subplots()

    ax.fill_between(agp["hour"], agp["p25"], agp["p75"], alpha=0.3)
    ax.plot(agp["hour"], agp["median"])

    ax.axhline(70, linestyle="--")
    ax.axhline(180, linestyle="--")

    ax.set_xlabel("Hora do dia")
    ax.set_ylabel("Glicemia (mg/dL)")
    ax.set_title("AGP simplificado – Dia médio")

    return fig


# =========================================================
# INTERFACE
# =========================================================
st.title("🩺 Insulin Pump Clinical Trainer")

# SESSION STATE — À PROVA DE CRASH
if "state" not in st.session_state:
    st.session_state.state = PhysiologyState()

if "history" not in st.session_state:
    st.session_state.history = []

if "step" not in st.session_state:
    st.session_state.step = 0

# SIDEBAR
pump_type = st.sidebar.selectbox(
    "Tipo de bomba",
    ["Convencional", "Suspensão automática", "Híbrido (AID)"]
)

days = st.sidebar.slider("Dias por consulta", 7, 30, 14)
basal = st.sidebar.slider("Basal (U/h)", 0.5, 2.0, 1.0, 0.1)
ic = st.sidebar.slider("IC (g/U)", 5, 20, 10)

delay = st.sidebar.slider("Atraso de bolus (min)", 0, 30, 15)
variability = st.sidebar.slider("Variabilidade", 0.0, 5.0, 1.0)

patient = PatientProfile(delay, variability)
pump = PumpSettings(basal, ic)

# STEP 0
if st.session_state.step == 0:
    if st.button("▶️ Rodar consulta"):
        state, df = simulate_consultation(
            st.session_state.state, patient, pump, pump_type, days
        )
        st.session_state.state = state
        st.session_state.history.append(df)
        st.session_state.step = 1

# STEP 1
if st.session_state.step == 1:
    df = st.session_state.history[-1]
    metrics = clinical_metrics(df)

    st.pyplot(plot_agp(df))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Glicemia média", metrics["mean"])
    c2.metric("TIR (%)", metrics["tir"])
    c3.metric("TBR (%)", metrics["tbr"])
    c4.metric("CV (%)", metrics["cv"])
