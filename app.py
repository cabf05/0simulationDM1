import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURAÇÃO
# =========================================================
st.set_page_config(page_title="CGM Clinical Simulator", layout="wide")

DT = 5                       # minutos
STEPS_PER_DAY = 1440 // DT   # 288

# =========================================================
# MODELOS
# =========================================================
class PhysiologyState:
    def __init__(self):
        self.glucose = 110.0
        self.insulin = np.zeros(4)
        self.carbs = np.zeros(2)


class PatientProfile:
    def __init__(self, variability):
        self.variability = variability
        self.isf = 40                      # mg/dL por U
        self.carb_abs = [0.03, 0.01]       # rápido / lento


class PumpSettings:
    def __init__(self, basal_profile, ic):
        self.basal_profile = basal_profile
        self.ic = ic


# =========================================================
# MOTOR FISIOLÓGICO
# =========================================================
def step_simulation(state, patient, pump, minute):
    hour = minute // 60

    # Basal
    basal_u = pump.basal_profile.get(hour, 1.0) * DT / 60
    state.insulin[0] += basal_u

    # Cinética da insulina
    transfer_rates = [0.25, 0.20, 0.15, 0.10]
    for i in range(3, 0, -1):
        moved = state.insulin[i - 1] * transfer_rates[i - 1]
        state.insulin[i] += moved
        state.insulin[i - 1] -= moved

    insulin_effect = state.insulin.sum() * patient.isf * DT / 240

    # Absorção de carboidrato
    carb_effect = 0
    for i in range(2):
        absorbed = state.carbs[i] * patient.carb_abs[i]
        carb_effect += absorbed
        state.carbs[i] -= absorbed

    # Variabilidade fisiológica REAL
    circadian = 6 * np.sin(2 * np.pi * hour / 24)
    noise = np.random.normal(0, patient.variability)

    state.glucose += carb_effect
    state.glucose -= insulin_effect
    state.glucose += 0.8
    state.glucose += circadian
    state.glucose += noise

    state.glucose = max(40, state.glucose)


# =========================================================
# SIMULAÇÃO
# =========================================================
def simulate(days, patient, pump):
    state = PhysiologyState()
    records = []

    base_meals = [(480, 60), (780, 70), (1140, 65)]

    for day in range(days):
        meals = [
            (t + np.random.randint(-20, 20),
             c + np.random.randint(-10, 10))
            for t, c in base_meals
        ]

        for step in range(STEPS_PER_DAY):
            minute = step * DT

            for meal_time, carbs in meals:
                if minute == meal_time:
                    state.carbs[0] += carbs * 0.7
                    state.carbs[1] += carbs * 0.3

                    bolus = (carbs / pump.ic) * np.random.normal(1, 0.15)
                    state.insulin[0] += bolus

            step_simulation(state, patient, pump, minute)

            records.append({
                "day": day,
                "minute": minute,
                "glucose": state.glucose
            })

    return pd.DataFrame(records)


# =========================================================
# AGP CLÍNICO (ROBUSTO)
# =========================================================
def plot_agp(df):
    agp = (
        df.groupby("minute")["glucose"]
        .agg(
            median=lambda x: np.percentile(x, 50),
            p10=lambda x: np.percentile(x, 10),
            p25=lambda x: np.percentile(x, 25),
            p75=lambda x: np.percentile(x, 75),
            p90=lambda x: np.percentile(x, 90),
        )
    )

    # garante eixo completo 0–1435
    full_index = np.arange(0, 1440, DT)
    agp = agp.reindex(full_index).interpolate()

    fig, ax = plt.subplots(figsize=(11, 4))

    ax.fill_between(full_index, agp["p25"], agp["p75"], alpha=0.45)
    ax.fill_between(full_index, agp["p10"], agp["p90"], alpha=0.25)
    ax.plot(full_index, agp["median"], linewidth=2)

    ax.axhline(70, linestyle="--")
    ax.axhline(180, linestyle="--")

    ax.set_xlim(0, 1440)
    ax.set_xlabel("Minutos do dia")
    ax.set_ylabel("Glicemia (mg/dL)")
    ax.set_title("AGP – Perfil Ambulatorial de Glicose")

    return fig


# =========================================================
# SESSION STATE (SEGUIMENTO)
# =========================================================
if "basal_profile" not in st.session_state:
    st.session_state.basal_profile = {h: 1.0 for h in range(24)}

if "ic" not in st.session_state:
    st.session_state.ic = 10


# =========================================================
# SIDEBAR – PRÓXIMA CONSULTA
# =========================================================
st.sidebar.header("Ajustes terapêuticos")

basal_night = st.sidebar.slider("Basal 0–6h (U/h)", 0.3, 2.5, 1.0, 0.1)
basal_day = st.sidebar.slider("Basal 6–24h (U/h)", 0.3, 2.5, 1.0, 0.1)

for h in range(24):
    st.session_state.basal_profile[h] = basal_night if h < 6 else basal_day

st.session_state.ic = st.sidebar.slider("IC (g/U)", 5, 20, st.session_state.ic)
variability = st.sidebar.slider("Variabilidade CGM", 2.0, 10.0, 6.0)

# =========================================================
# EXECUÇÃO
# =========================================================
st.title("Simulador Clínico de CGM")

if st.button("▶️ Rodar consulta"):
    patient = PatientProfile(variability)
    pump = PumpSettings(st.session_state.basal_profile, st.session_state.ic)

    df = simulate(days=14, patient=patient, pump=pump)

    st.pyplot(plot_agp(df))
