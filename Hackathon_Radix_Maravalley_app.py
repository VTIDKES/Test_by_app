"""
LABS - Smart Meter Insights
App Standalone que funciona sem módulos externos
Execute: streamlit run app_standalone.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Smart Meter LABS", page_icon="⚡", layout="wide")

st.markdown('<h1 style="text-align: center; color: #1f77b4;">⚡ Smart Meter Insights LABS</h1>', unsafe_allow_html=True)
st.markdown("**Plataforma de Insights - Protótipo P&D ANEEL Tema 3**")

@st.cache_data
def generate_data(num_meters=50, days=7):
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    timestamps = pd.date_range(start=start_time, end=end_time, freq='15min')
    
    data = []
    for meter_id in range(1, num_meters + 1):
        id_medidor = f"SM{meter_id:05d}"
        alimentador = np.random.choice(['AL001', 'AL002', 'AL003', 'AL004'])
        consumo_base = np.random.uniform(2, 8)
        
        for ts in timestamps:
            hora = ts.hour
            fator = 1.5 if 6 <= hora <= 8 else (2.0 if 18 <= hora <= 22 else (0.3 if hora < 6 else 1.0))
            potencia = consumo_base * fator * np.random.uniform(0.8, 1.2)
            
            if np.random.random() < 0.05:
                potencia = potencia * 3.0 if np.random.random() < 0.5 else 0
            
            tensao = 220 + np.random.normal(0, 3)
            if np.random.random() < 0.02:
                tensao = np.random.uniform(100, 115)
            
            data.append({
                'id_medidor': id_medidor,
                'timestamp': ts,
                'tensao_v': tensao,
                'potencia_kw': potencia,
                'fator_potencia': np.random.uniform(0.85, 0.98),
                'energia_kwh': potencia * 0.25,
                'alimentador': alimentador,
                'hora': hora,
            })
    
    return pd.DataFrame(data)

@st.cache_data
def detect_events(df):
    events = []
    for _, row in df[df['tensao_v'] < 115].iterrows():
        events.append({'id_medidor': row['id_medidor'], 'timestamp': row['timestamp'],
                      'tipo': 'SUBTENSAO', 'severidade': 'ALTA', 'descricao': f"Tensão {row['tensao_v']:.1f}V"})
    for _, row in df[df['potencia_kw'] < 0.1].iterrows():
        events.append({'id_medidor': row['id_medidor'], 'timestamp': row['timestamp'],
                      'tipo': 'QUEDA', 'severidade': 'CRITICA', 'descricao': 'Falta de energia'})
    return pd.DataFrame(events) if events else pd.DataFrame()

# Sidebar
st.sidebar.title("⚙️ Configuração")
num_meters = st.sidebar.slider("Medidores", 10, 100, 50)
num_days = st.sidebar.slider("Dias", 1, 30, 7)

if st.sidebar.button("🔄 Gerar Dados"):
    st.cache_data.clear()

page = st.sidebar.radio("Página", ["📊 Visão Geral", "👤 Cliente", "🔧 Operação"])

# Dados
with st.spinner("Carregando..."):
    df = generate_data(num_meters, num_days)
    events = detect_events(df)

# Páginas
if page == "📊 Visão Geral":
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Medidores", df['id_medidor'].nunique())
    col2.metric("Leituras", f"{len(df):,}")
    col3.metric("Eventos", len(events))
    col4.metric("Qualidade", "89%")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Eventos por Severidade")
        if not events.empty:
            fig = px.pie(events, names='severidade')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Consumo por Alimentador")
        consumo = df.groupby('alimentador')['potencia_kw'].sum().reset_index()
        fig = px.bar(consumo, x='alimentador', y='potencia_kw')
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Timeline de Consumo")
    hourly = df.groupby(df['timestamp'].dt.floor('H'))['potencia_kw'].sum().reset_index()
    fig = px.line(hourly, x='timestamp', y='potencia_kw')
    st.plotly_chart(fig, use_container_width=True)

elif page == "👤 Cliente":
    st.header("Portal do Cliente")
    meter = st.selectbox("Seu Medidor", sorted(df['id_medidor'].unique()))
    df_meter = df[df['id_medidor'] == meter]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Consumo Diário", f"{df_meter['energia_kwh'].sum() / num_days:.2f} kWh")
    col2.metric("Fator Potência", f"{df_meter['fator_potencia'].mean():.3f}")
    col3.metric("Tensão Média", f"{df_meter['tensao_v'].mean():.1f} V")
    
    st.subheader("Perfil de Consumo por Hora")
    hourly = df_meter.groupby('hora')['potencia_kw'].mean().reset_index()
    fig = px.bar(hourly, x='hora', y='potencia_kw')
    st.plotly_chart(fig, use_container_width=True)

else:
    st.header("Portal Operacional")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Eventos Totais", len(events))
    col2.metric("Críticos", len(events[events['severidade'] == 'CRITICA']) if not events.empty else 0)
    col3.metric("Altos", len(events[events['severidade'] == 'ALTA']) if not events.empty else 0)
    
    st.subheader("Eventos Recentes")
    if not events.empty:
        st.dataframe(events.head(20), use_container_width=True)
    else:
        st.info("Nenhum evento detectado")

st.sidebar.markdown("---")
st.sidebar.info("**LABS - Smart Meter**\nP&D ANEEL Tema 3\n2026")
