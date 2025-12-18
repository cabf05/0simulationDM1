import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import io
from datetime import datetime, timedelta

# --- CONFIGURAÇÃO DA PÁGINA (Deve ser o primeiro comando Streamlit) ---
st.set_page_config(
    page_title="Simulador Clínico de Bomba de Insulina",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="💉"
)

# --- CLASSES E MOTOR DE SIMULAÇÃO (Backend Lógico) ---

class PatientProfile:
    def __init__(self, name, weight, tdd, basal_profile, icr, isf, target, pump_mode, bolus_delay=0):
        self.name = name
        self.weight = weight
        self.tdd = tdd  # Total Daily Dose
        self.basal_profile = basal_profile  # Dict {hour (int): rate (float)}
        self.icr = icr  # Insulin Carb Ratio (g/U)
        self.isf = isf  # Insulin Sensitivity Factor (mg/dL/U)
        self.target = target  # Target Glucose
        self.pump_mode = pump_mode  # 'Open Loop', 'PLGS', 'AID (Híbrida)'
        self.bolus_delay = bolus_delay  # Minutes delay relative to meal (negative = pre-bolus)
        
    def to_dict(self):
        # Retorna uma cópia rasa do dicionário para evitar modificação de referência
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

class SimulationEngine:
    def __init__(self):
        # Constantes Farmacocinéticas (Simplificadas para Educação)
        self.ins_action_time = 240  # 4 horas em minutos
        self.carb_action_time = 180 # 3 horas em minutos
        self.dt = 5 # Passo de tempo em minutos

    def get_insulin_activity(self, time_since_injection):
        # Curva de ação simplificada (Pico em ~75min)
        t = time_since_injection
        if t < 0 or t > self.ins_action_time: return 0
        return (t * np.exp(-t / 75)) 

    def get_carb_absorption(self, time_since_meal):
        # Curva de absorção (Pico em ~60min)
        t = time_since_meal
        if t < 0 or t > self.carb_action_time: return 0
        return (t * np.exp(-t / 60))

    def run_simulation(self, patient, days, start_glucose=120, history=None):
        """
        Executa a simulação determinística.
        """
        n_steps = int((days * 24 * 60) / self.dt)
        
        # Estruturas de dados
        times = []
        glucose = []
        basal_delivered = []
        bolus_delivered = []
        carbs_eaten = []
        
        current_g = start_glucose
        
        # Se houver histórico, pegar o último estado
        if history is not None and not history.empty:
            current_g = history['Glucose'].iloc[-1]
            start_date = history['Time'].iloc[-1] + timedelta(minutes=self.dt)
        else:
            start_date = datetime(2024, 1, 1, 0, 0, 0)

        # Buffers para efeito acumulado (Stacks)
        active_insulin_stack = [] # Tuplas (amount, time_injected)
        active_carb_stack = []    # Tuplas (amount, time_eaten)

        # Padrão Alimentar Fixo (Determinístico)
        meal_schedule = {
            8:  {'carbs': 40, 'type': 'Café'},
            13: {'carbs': 70, 'type': 'Almoço'},
            20: {'carbs': 60, 'type': 'Jantar'}
        }

        current_time = start_date

        for step in range(n_steps):
            hour = current_time.hour
            minute = current_time.minute
            
            # 1. Definir Basal Base (Programado)
            # Tenta pegar a hora como int, fallback para 0.5 se erro
            base_basal = patient.basal_profile.get(hour, patient.basal_profile.get(str(hour), 0.5))
            
            # 2. Lógica da Bomba (Modulação do Basal)
            actual_basal = base_basal
            
            if patient.pump_mode == "PLGS": # Suspensão preditiva (Simplificada para Low Glucose Suspend)
                if current_g < 80:
                    actual_basal = 0
            
            elif patient.pump_mode == "AID (Híbrida)": # Loop Fechado Simplificado
                deviation = current_g - patient.target
                # Algoritmo P simples
                factor = deviation / patient.isf 
                adjustment = (factor * 0.5) if deviation > 0 else (factor * 0.8)
                actual_basal = max(0, min(base_basal * 3, base_basal + adjustment))

            # Entrega Basal neste step (Unidades)
            basal_step = (actual_basal / 60) * self.dt
            active_insulin_stack.append({'amt': basal_step, 'time': 0})
            
            # 3. Refeições e Bolus
            carb_input = 0
            bolus_step = 0
            
            # Inserir Carbo na hora da refeição
            if minute < self.dt: 
                if hour in meal_schedule:
                    meal = meal_schedule[hour]
                    carb_input = meal['carbs']
                    active_carb_stack.append({'amt': carb_input, 'time': 0})

            # Lógica de Bolus com Delay Comportamental
            # Verifica se "agora" é o momento de aplicar o bolus de alguma refeição
            for m_hour, m_data in meal_schedule.items():
                meal_time_min_day = m_hour * 60
                current_time_min_day = hour * 60 + minute
                
                # O momento do bolus é: Hora da Comida + Delay do Paciente
                scheduled_bolus_time = meal_time_min_day + patient.bolus_delay
                
                # Se o tempo atual coincide com o tempo agendado do bolus (dentro do intervalo dt)
                # Nota: Isso lida com mudança de dia simples (não trata virada meia noite complexa p/ simplificar)
                if abs(current_time_min_day - scheduled_bolus_time) < (self.dt / 2.0):
                     carb_ref = m_data['carbs']
                     
                     # Cálculo do Bolus
                     b_calc = carb_ref / patient.icr
                     # Correção baseada na glicemia DO MOMENTO DA APLICAÇÃO
                     b_corr = max(0, (current_g - patient.target) / patient.isf)
                     
                     bolus_step = b_calc + b_corr
                     active_insulin_stack.append({'amt': bolus_step, 'time': 0})

            # 4. Dinâmica Fisiológica (Atualizar Stacks e Efeitos)
            
            # Insulina
            insulin_effect = 0
            new_ins_stack = []
            for item in active_insulin_stack:
                item['time'] += self.dt
                activity = self.get_insulin_activity(item['time'])
                norm_activity = activity / 5625 # Normalização empírica da área
                
                effect = item['amt'] * norm_activity * patient.isf 
                insulin_effect += effect
                
                if item['time'] < self.ins_action_time:
                    new_ins_stack.append(item)
            active_insulin_stack = new_ins_stack

            # Carboidratos
            carb_effect = 0
            new_carb_stack = []
            for item in active_carb_stack:
                item['time'] += self.dt
                activity = self.get_carb_absorption(item['time'])
                norm_activity = activity / 3600 # Normalização empírica
                
                # Conversão ISF/ICR
                carb_rise_factor = patient.isf / patient.icr
                
                effect = item['amt'] * norm_activity * carb_rise_factor
                carb_effect += effect
                
                if item['time'] < self.carb_action_time:
                    new_carb_stack.append(item)
            active_carb_stack = new_carb_stack
            
            # Drift Hepático (Endógeno)
            endogenous_rise = (0.5 * patient.isf / 60) * self.dt 
            
            # Atualização da Glicemia
            delta_g = endogenous_rise + carb_effect - insulin_effect
            
            # Ruído metabólico mínimo (Ciclo circadiano)
            circadian = np.sin((hour - 6) * np.pi / 12) * 2 * (self.dt/60)
            
            current_g += delta_g + circadian
            current_g = max(40, current_g) # Clamp mínimo

            # Gravar dados
            times.append(current_time)
            glucose.append(int(current_g))
            basal_delivered.append(actual_basal)
            bolus_delivered.append(bolus_step)
            carbs_eaten.append(carb_input if minute < self.dt else 0)
            
            current_time += timedelta(minutes=self.dt)

        # Criar DataFrame
        df_result = pd.DataFrame({
            'Time': times,
            'Glucose': glucose,
            'Basal': basal_delivered,
            'Bolus': bolus_delivered,
            'Carbs': carbs_eaten
        })
        
        return df_result

# --- FUNÇÕES DE AUXÍLIO E ESTADO ---

def init_session_state():
    if 'patients' not in st.session_state:
        # Perfil 1: Convencional, Aderente
        p1 = PatientProfile("Paciente A (Padrão)", 70, 40, {h: 0.8 for h in range(24)}, 15, 40, 110, "Open Loop", 0)
        # Perfil 2: Atraso no Bolus
        p2 = PatientProfile("Paciente B (Atraso Bolus)", 70, 40, {h: 0.8 for h in range(24)}, 15, 40, 110, "Open Loop", 30)
        
        st.session_state.patients = [p1, p2]
        st.session_state.current_patient_idx = 0
        st.session_state.simulation_history = pd.DataFrame()

def save_simulation(patient, history):
    # CORREÇÃO: Cria uma cópia dos parâmetros para não alterar o objeto em memória
    params = patient.to_dict()
    
    # Converter basal profile para string JSON apenas para a exportação
    if isinstance(params['basal_profile'], dict):
        params['basal_profile'] = json.dumps(params['basal_profile'])
    
    meta_json = json.dumps(params)
    csv_buffer = io.StringIO()
    
    # Escrever metadados como comentário na primeira linha
    csv_buffer.write(f"#METADATA:{meta_json}\n")
    history.to_csv(csv_buffer, index=False)
    
    return csv_buffer.getvalue()

def load_simulation(uploaded_file):
    try:
        content = uploaded_file.getvalue().decode("utf-8")
        lines = content.splitlines()
        first_line = lines[0]
        
        if first_line.startswith("#METADATA:"):
            meta_json = first_line.replace("#METADATA:", "")
            params = json.loads(meta_json)
            
            # CORREÇÃO: Reconstrói o dicionário de basal com chaves inteiras
            if isinstance(params['basal_profile'], str):
                raw_profile = json.loads(params['basal_profile'])
                params['basal_profile'] = {int(k): v for k,v in raw_profile.items()}
            
            patient = PatientProfile.from_dict(params)
            
            # Carregar CSV
            data = pd.read_csv(io.StringIO("\n".join(lines[1:])))
            data['Time'] = pd.to_datetime(data['Time'])
            
            return patient, data
        else:
            st.error("Formato de arquivo inválido: Metadados ausentes.")
            return None, None
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {str(e)}")
        return None, None

def calculate_metrics(df):
    if df.empty: return {}
    g = df['Glucose']
    total = len(g)
    if total == 0: return {}
    
    tir = len(g[(g >= 70) & (g <= 180)]) / total * 100
    tbr = len(g[g < 70]) / total * 100
    tar = len(g[g > 180]) / total * 100
    mean_g = g.mean()
    gmi = 3.31 + (0.02392 * mean_g)
    cv = (g.std() / mean_g) * 100 if mean_g > 0 else 0
    
    return {
        "TIR": tir, "TBR": tbr, "TAR": tar, 
        "Mean": mean_g, "GMI": gmi, "CV": cv
    }

# --- INTERFACE GRÁFICA ---

init_session_state()

st.title("Sistema de Simulação Clínica: Bomba de Insulina")
st.markdown("---")

# SIDEBAR: Controle Global
with st.sidebar:
    st.header("🗂️ Gestão de Simulação")
    
    # Seletor de Paciente
    patient_names = [p.name for p in st.session_state.patients]
    selected_p_idx = st.selectbox("Selecionar Paciente", range(len(patient_names)), format_func=lambda x: patient_names[x])
    
    # Atualizar paciente atual
    st.session_state.current_patient_idx = selected_p_idx
    current_patient = st.session_state.patients[selected_p_idx]

    st.markdown("---")
    st.subheader("💾 Persistência")
    
    # Exportar
    if not st.session_state.simulation_history.empty:
        csv_data = save_simulation(current_patient, st.session_state.simulation_history)
        st.download_button(
            label="📥 Baixar Simulação (CSV)",
            data=csv_data,
            file_name=f"sim_{current_patient.name.replace(' ', '_')}.csv",
            mime="text/csv",
            help="Salva parâmetros e histórico clínico."
        )

    # Importar
    uploaded_file = st.file_uploader("📤 Carregar Simulação", type=["csv"])
    if uploaded_file is not None:
        if st.button("Restaurar Simulação"):
            p, d = load_simulation(uploaded_file)
            if p:
                st.session_state.patients.append(p)
                st.session_state.current_patient_idx = len(st.session_state.patients) - 1
                st.session_state.simulation_history = d
                st.success("Simulação carregada! O novo perfil foi adicionado à lista.")
                st.rerun()

# TABS PRINCIPAIS
tab1, tab2, tab3 = st.tabs(["⚙️ Parâmetros Clínicos", "⏱️ Executar Simulação", "📊 Análise Clínica"])

# --- TAB 1: PARÂMETROS ---
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configuração do Paciente")
        new_name = st.text_input("Nome do Perfil", current_patient.name)
        current_patient.name = new_name
        
        current_patient.weight = st.number_input("Peso (kg)", value=float(current_patient.weight))
        
        st.subheader("Configuração da Bomba")
        pump_options = ["Open Loop", "PLGS", "AID (Híbrida)"]
        
        # Garante que o modo atual esteja na lista
        curr_mode_index = 0
        if current_patient.pump_mode in pump_options:
            curr_mode_index = pump_options.index(current_patient.pump_mode)
            
        current_patient.pump_mode = st.selectbox("Modo de Operação", pump_options, index=curr_mode_index)
        current_patient.target = st.number_input("Meta Glicêmica (mg/dL)", value=int(current_patient.target), step=5)

    with col2:
        st.subheader("Parâmetros de Insulina")
        c1, c2 = st.columns(2)
        with c1:
            current_patient.icr = st.number_input("Relação Insulina/Carb (1:X)", value=float(current_patient.icr), min_value=1.0)
            current_patient.isf = st.number_input("Fator de Sensibilidade (1:X)", value=float(current_patient.isf), min_value=1.0)
        with c2:
            current_patient.bolus_delay = st.slider("Comportamento: Atraso no Bolus (min)", -30, 60, value=int(current_patient.bolus_delay), step=5, help="Negativo = Pré-bolus. Positivo = Esquecimento/Atraso.")

        st.markdown("#### Perfil Basal (U/h)")
        
        # Recuperação segura do basal da hora 0
        basal_h0 = current_patient.basal_profile.get(0, 0.5)
        
        basal_val = st.number_input("Basal Global (Simplificado)", value=float(basal_h0), step=0.05)
        
        if st.button("Aplicar Basal Global a todas as horas"):
            current_patient.basal_profile = {h: basal_val for h in range(24)}
            st.success("Perfil basal atualizado para todas as 24h.")
            
        with st.expander("Ver Perfil Basal Detalhado (JSON)"):
            st.json(current_patient.basal_profile)

# --- TAB 2: SIMULAÇÃO ---
with tab2:
    st.markdown("### ⏩ Consultório Virtual")
    st.info("O tempo avança de forma acelerada. Cada execução adiciona dias ao histórico do paciente.")
    
    days_to_sim = st.slider("Tempo até retorno (dias)", 1, 30, 14)
    
    if st.button("Simular Período e Gerar Dados", type="primary"):
        engine = SimulationEngine()
        with st.spinner("Calculando dinâmica farmacocinética..."):
            new_data = engine.run_simulation(
                current_patient, 
                days_to_sim, 
                history=st.session_state.simulation_history
            )
            
            # Concatenar histórico mantendo coerência temporal
            if st.session_state.simulation_history.empty:
                st.session_state.simulation_history = new_data
            else:
                st.session_state.simulation_history = pd.concat([st.session_state.simulation_history, new_data]).drop_duplicates(subset=['Time']).sort_values('Time')
            
        st.success(f"Simulação concluída! {days_to_sim} dias adicionados.")

# --- TAB 3: ANÁLISE ---
with tab3:
    df = st.session_state.simulation_history
    
    if df.empty:
        st.warning("Nenhum dado clínico disponível. Execute uma simulação na aba anterior.")
    else:
        # Métricas
        metrics = calculate_metrics(df)
        
        st.subheader("📋 Painel Clínico Consolidado")
        m1, m2, m3, m4, m5 = st.columns(5)
        if metrics:
            m1.metric("Média Glicêmica", f"{metrics['Mean']:.0f} mg/dL")
            m2.metric("GMI (A1c Est)", f"{metrics['GMI']:.1f} %")
            m3.metric("Tempo no Alvo (TIR)", f"{metrics['TIR']:.1f} %", delta_color="normal" if metrics['TIR']>70 else "inverse")
            m4.metric("Hipoglicemia (<70)", f"{metrics['TBR']:.1f} %", delta_color="inverse")
            m5.metric("CV (Variabilidade)", f"{metrics['CV']:.1f} %")

        st.markdown("---")

        # AGP - Ambulatory Glucose Profile
        st.subheader("📈 Perfil Ambulatorial de Glicose (AGP)")
        
        
        # Prepara dados para AGP
        df['HourOfDay'] = df['Time'].dt.hour + df['Time'].dt.minute/60.0
        df['TimeBin'] = (df['HourOfDay'] * 4).astype(int) / 4.0 # Bins de 15 min
        
        agp_data = df.groupby('TimeBin')['Glucose'].agg([
            'median', 
            lambda x: x.quantile(0.25), 
            lambda x: x.quantile(0.75), 
            lambda x: x.quantile(0.10), 
            lambda x: x.quantile(0.90)
        ]).reset_index()
        agp_data.columns = ['Time', 'Median', 'Q25', 'Q75', 'Q10', 'Q90']
        
        fig_agp = go.Figure()
        
        # Faixas de percentil (Sombreamento)
        fig_agp.add_trace(go.Scatter(x=agp_data['Time'], y=agp_data['Q90'], mode='lines', line=dict(width=0), showlegend=False, name='Q90'))
        fig_agp.add_trace(go.Scatter(x=agp_data['Time'], y=agp_data['Q10'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(200, 200, 200, 0.3)', showlegend=False, name='Q10-90'))
        
        fig_agp.add_trace(go.Scatter(x=agp_data['Time'], y=agp_data['Q75'], mode='lines', line=dict(width=0), showlegend=False, name='Q75'))
        fig_agp.add_trace(go.Scatter(x=agp_data['Time'], y=agp_data['Q25'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 100, 80, 0.4)', showlegend=False, name='Q25-75'))
        
        # Mediana
        fig_agp.add_trace(go.Scatter(x=agp_data['Time'], y=agp_data['Median'], mode='lines', line=dict(color='black', width=2), name='Mediana'))
        
        # Faixas alvo
        fig_agp.add_hrect(y0=70, y1=180, line_width=0, fillcolor="green", opacity=0.1)
        fig_agp.add_hline(y=70, line_dash="dot", line_color="red")
        fig_agp.add_hline(y=180, line_dash="dot", line_color="orange")
        
        fig_agp.update_layout(
            title="Dia Modal (24h)",
            xaxis_title="Hora do Dia",
            yaxis_title="Glicose (mg/dL)",
            xaxis=dict(tickmode='array', tickvals=[0,6,12,18,24], ticktext=['00:00','06:00','12:00','18:00','24:00']),
            yaxis=dict(range=[0, 400]),
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_agp, use_container_width=True)

        # Visualização Longitudinal
        st.subheader("📅 Histórico Longitudinal")
        fig_long = px.line(df, x='Time', y='Glucose', title="Evolução Temporal Completa")
        fig_long.add_hrect(y0=70, y1=180, line_width=0, fillcolor="green", opacity=0.1)
        st.plotly_chart(fig_long, use_container_width=True)
        
        # Dados Brutos
        with st.expander("Ver Dados Brutos"):
            st.dataframe(df)

# --- RODAPÉ EDUCACIONAL ---
st.markdown("---")
st.caption("Sistema de Simulação Educacional - Diabetes Tipo 1. Ferramenta didática determinística.")
