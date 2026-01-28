"""
CPFL LABS | TEMA 3
Smart Meter Insights Platform (Python Prototype)
Interface profissional baseada no design CPFL
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

# Configuração da página
st.set_page_config(
    page_title="CPFL LABS | TEMA 3",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado - Estilo CPFL
st.markdown("""
<style>
    /* Header principal */
    .main-header {
        background: linear-gradient(90deg, #0A5F7F 0%, #0D8AB5 100%);
        color: white;
        padding: 1.5rem;
        margin: -1rem -1rem 2rem -1rem;
        border-bottom: 3px solid #00A9CE;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.3rem 0 0 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F8F9FA;
        border-right: 1px solid #DEE2E6;
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        font-weight: 600;
        color: #495057;
    }
    
    /* Métricas */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #0A5F7F;
    }
    
    /* Cards de seção */
    .section-card {
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border-left: 4px solid #00A9CE;
    }
    
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #0A5F7F;
        margin-bottom: 1rem;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .badge-normal { background: #D1F2EB; color: #0C7C59; }
    .badge-alerta { background: #FFF3CD; color: #856404; }
    .badge-critico { background: #F8D7DA; color: #721C24; }
    
    /* Botões */
    .stButton > button {
        background: #00A9CE;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background: #0A5F7F;
    }
    
    /* Info box */
    .info-box {
        background: #E7F3F7;
        border-left: 4px solid #00A9CE;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    /* Terminal style */
    .terminal-output {
        background: #1E1E1E;
        color: #D4D4D4;
        font-family: 'Courier New', monospace;
        padding: 1rem;
        border-radius: 4px;
        font-size: 0.85rem;
        margin: 1rem 0;
    }
    
    /* Status footer */
    .status-footer {
        background: #F8F9FA;
        padding: 0.8rem;
        margin-top: 2rem;
        border-top: 1px solid #DEE2E6;
        font-size: 0.85rem;
        color: #6C757D;
    }
</style>
""", unsafe_allow_html=True)

# Funções de geração de dados
@st.cache_data
def generate_smart_meter_data(num_meters=50, days=7):
    """Gera dados sintéticos de smart meters"""
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    timestamps = pd.date_range(start=start_time, end=end_time, freq='15min')
    
    alimentadores = ['AL-04', 'AL-05', 'AL-06', 'AL-07']
    data = []
    
    for meter_id in range(1, num_meters + 1):
        id_medidor = f"MEU-{39200 + meter_id}"
        alimentador = np.random.choice(alimentadores)
        consumo_base = np.random.uniform(1.5, 3.5)
        
        for ts in timestamps:
            hora = ts.hour
            # Padrão residencial
            if 6 <= hora <= 8 or 18 <= hora <= 22:
                fator = 2.5
            elif 0 <= hora <= 6:
                fator = 0.4
            else:
                fator = 1.0
            
            potencia = consumo_base * fator * np.random.uniform(0.85, 1.15)
            
            # Anomalias raras
            if np.random.random() < 0.02:
                potencia = potencia * 4.0 if np.random.random() < 0.5 else 0.05
            
            tensao = 127 + np.random.normal(0, 2)
            if np.random.random() < 0.01:
                tensao = np.random.uniform(110, 117)
            
            fator_pot = np.random.uniform(0.88, 0.98)
            if np.random.random() < 0.02:
                fator_pot = np.random.uniform(0.65, 0.75)
            
            data.append({
                'id_medidor': id_medidor,
                'timestamp': ts,
                'tensao_v': tensao,
                'potencia_kw': potencia,
                'fator_potencia': fator_pot,
                'energia_kwh': potencia * 0.25,
                'alimentador': alimentador,
                'hora': hora,
            })
    
    return pd.DataFrame(data)

@st.cache_data
def detect_events_advanced(df):
    """Detecção avançada de eventos"""
    events = []
    event_id = 1
    
    for _, row in df.iterrows():
        # Subtensão
        if row['tensao_v'] < 117:
            events.append({
                'id_evento': f"EVT-{event_id:05d}",
                'id_medidor': row['id_medidor'],
                'timestamp': row['timestamp'],
                'tipo': 'SUBTENSÃO',
                'severidade': 'CRÍTICA' if row['tensao_v'] < 110 else 'ALTA',
                'valor': f"{row['tensao_v']:.1f}V",
                'descricao': f"Tensão abaixo do adequado ({row['tensao_v']:.1f}V)",
                'acao_sugerida': 'Verificar rede e transformador',
                'destino': 'Operação'
            })
            event_id += 1
        
        # Queda de energia
        if row['potencia_kw'] < 0.1:
            events.append({
                'id_evento': f"EVT-{event_id:05d}",
                'id_medidor': row['id_medidor'],
                'timestamp': row['timestamp'],
                'tipo': 'QUEDA DE ENERGIA',
                'severidade': 'CRÍTICA',
                'valor': f"{row['potencia_kw']:.2f}kW",
                'descricao': 'Possível interrupção no fornecimento',
                'acao_sugerida': 'Despachar equipe imediatamente',
                'destino': 'Operação'
            })
            event_id += 1
        
        # Fator de potência baixo
        if row['fator_potencia'] < 0.75:
            events.append({
                'id_evento': f"EVT-{event_id:05d}",
                'id_medidor': row['id_medidor'],
                'timestamp': row['timestamp'],
                'tipo': 'FP BAIXO',
                'severidade': 'MÉDIA',
                'valor': f"{row['fator_potencia']:.2f}",
                'descricao': f"Fator de potência inadequado ({row['fator_potencia']:.2f})",
                'acao_sugerida': 'Notificar cliente sobre correção',
                'destino': 'Cliente'
            })
            event_id += 1
    
    return pd.DataFrame(events) if events else pd.DataFrame()

# Header da aplicação
st.markdown("""
<div class="main-header">
    <h1>🔷 CPFL LABS | TEMA 3</h1>
    <p>Smart Meter Insights Platform (Python Prototype)</p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Módulos do Pipeline
with st.sidebar:
    st.markdown("### MÓDULOS DO PIPELINE")
    
    page = st.radio(
        "",
        [
            "📊 Ingestão & Qualidade",
            "📈 Visão Operacional",
            "🔍 Análise Avançada",
            "⚡ Motor de Eventos",
            "🔧 GIS / ADMS / SAP"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("### Configurações")
    num_meters = st.slider("Medidores", 10, 100, 50, 10)
    num_days = st.slider("Dias", 1, 30, 7)
    
    if st.button("🔄 Atualizar Dados"):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.8rem; color: #6C757D;'>
    <strong>Status do Laboratório:</strong><br>
    Ambiente de teste (Sandbox)<br>
    Dados sintéticos carregados<br><br>
    <strong>Versões:</strong><br>
    🟢 KERNEL: PYTHON 3.10<br>
    🟢 PANDAS 2.0.1<br>
    🟢 NUMPY 1.24<br>
    🟢 STREAMLIT MOCK
    </div>
    """, unsafe_allow_html=True)

# Carregar dados
with st.spinner("Carregando dados..."):
    df = generate_smart_meter_data(num_meters, num_days)
    events_df = detect_events_advanced(df)

# Roteamento de páginas
if page == "📊 Ingestão & Qualidade":
    st.markdown('<div class="section-title">Pipeline de Ingestão de Dados (MDC/MDM)</div>', unsafe_allow_html=True)
    st.markdown("Simulação da leitura de arquivos brutos, limpeza e cálculo de score de confiabilidade.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("▶️ Executar Pipeline ETL", use_container_width=True):
            with st.spinner("Processando..."):
                st.markdown("""
                <div class="terminal-output">
                user@cpfl-labs:~$ python run_pipeline.py --source mdm --validate<br><br>
                [01:31:44] INFO: Inicializando Ingestão de Dados MDM...<br>
                [01:31:44] INFO: Conectando ao Data Lake (S3 Mock)... OK<br>
                [01:31:45] INFO: Lendo arquivo raw/mdm_export_2026.csv...<br>
                [01:31:45] INFO: Schema Detectado: [meter_id, timestamp, v_a, i_a, kw_tot, kvar_tot]<br>
                [01:31:46] INFO: Validando """ + f"{len(df)}" + """ registros...<br>
                [01:31:47] INFO: Aviso: """ + f"{int(len(df) * 0.002)}" + """ registros com gaps de timestamp<br>
                [01:31:47] INFO: Calculando Estatísticas Básicas...<br>
                [01:31:48] INFO: Verificando limites PRODIST Módulo 8...<br>
                [01:31:48] INFO: Gerando Features: [peak_load, voltage_violation_index, theft_score]<br>
                [01:31:49] INFO: Persistindo dados processados em data/refined/smart_meter_data.parquet<br>
                [01:31:50] INFO: Ingestão concluída com sucesso. Tempo total: 1.2s
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
        📁 Origem: /data/raw/mdm_export_2026.csv
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Métricas de qualidade
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.metric("REGISTROS PROCESSADOS", f"{len(df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        missing = int(len(df) * 0.002)
        st.metric("DADOS FALTANTES (CORRIGIDOS)", missing)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.metric("SCORE DE CONFIABILIDADE MÉDIO", "99.4%")
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "📈 Visão Operacional":
    st.markdown('<div class="section-title">Visão Operacional da Rede</div>', unsafe_allow_html=True)
    st.markdown("Monitoramento em tempo real do Alimentador AL-04 (Campinas/SP)")
    
    # Métricas principais
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tensao_media = df['tensao_v'].mean()
        st.metric("Tensão Média", f"{tensao_media:.1f} V", 
                 delta=f"{tensao_media - 127:.1f}V", 
                 delta_color="off")
    
    with col2:
        eventos_criticos = len(events_df[events_df['severidade'] == 'CRÍTICA']) if not events_df.empty else 0
        st.metric("Eventos Críticos", eventos_criticos,
                 delta="2 novos" if eventos_criticos > 0 else "0",
                 delta_color="inverse")
    
    with col3:
        carga_total = df['potencia_kw'].sum() / 1000
        st.metric("Carga Total", f"{carga_total:.1f} MW")
    
    st.markdown("---")
    
    # Mapa (simulado)
    st.markdown('<div class="section-title">🗺️ Geolocalização de Ativos</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("🗺️ Mapa interativo: Alimentador AL-04 (Campinas/SP) com marcadores de medidores e eventos")
        st.markdown("""
        <div style='background: #E8F4F8; padding: 2rem; border-radius: 8px; text-align: center;'>
        <strong>Visualização GIS</strong><br>
        📍 """ + str(num_meters) + """ medidores ativos<br>
        🟢 """ + str(num_meters - (eventos_criticos if eventos_criticos else 0)) + """ Normal  
        🔴 """ + str(eventos_criticos) + """ Com alerta
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Eventos Prioritários")
        if not events_df.empty:
            priority_events = events_df[events_df['severidade'].isin(['CRÍTICA', 'ALTA'])].head(3)
            for _, evt in priority_events.iterrows():
                badge_class = "badge-critico" if evt['severidade'] == 'CRÍTICA' else "badge-alerta"
                st.markdown(f"""
                <div class="section-card">
                <span class="status-badge {badge_class}">{evt['severidade']}</span><br>
                <strong>{evt['tipo']}</strong><br>
                {evt['id_medidor']}<br>
                <small>{evt['descricao']}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ Nenhum evento prioritário")
    
    st.markdown("---")
    
    # Balanço Energético
    st.markdown('<div class="section-title">Balanço Energético (Últimas 24h)</div>', unsafe_allow_html=True)
    
    last_24h = df[df['timestamp'] >= (datetime.now() - timedelta(hours=24))]
    hourly = last_24h.groupby(last_24h['timestamp'].dt.hour).agg({
        'potencia_kw': 'sum',
        'energia_kwh': 'sum'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hourly['timestamp'],
        y=hourly['energia_kwh'],
        name='Consumo (kWh)',
        marker_color='#FFB300'
    ))
    fig.update_layout(
        height=350,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="Hora",
        yaxis_title="Energia (kWh)",
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "🔍 Análise Avançada":
    st.markdown('<div class="section-title">Análise Avançada</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Selecionar Medidor**")
        selected_meter = st.selectbox("", sorted(df['id_medidor'].unique()), label_visibility="collapsed")
        
        st.markdown("**Período**")
        period = st.selectbox("", ["Últimas 24 Horas", "Últimos 7 Dias", "Últimos 30 Dias"], label_visibility="collapsed")
        
        if st.button("🔄 Atualizar Análise", use_container_width=True):
            st.rerun()
    
    with col2:
        st.markdown(f"**Medidor: {selected_meter}**")
        
        meter_data = df[df['id_medidor'] == selected_meter]
        
        if not meter_data.empty:
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Consumo Médio", f"{meter_data['energia_kwh'].mean():.2f} kWh")
            col_b.metric("FP Médio", f"{meter_data['fator_potencia'].mean():.3f}")
            col_c.metric("Tensão Média", f"{meter_data['tensao_v'].mean():.1f} V")
    
    st.markdown("---")
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-title">Perfil de Tensão (PRODIST Módulo 8)</div>', unsafe_allow_html=True)
        
        meter_data_sorted = meter_data.sort_values('timestamp')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=meter_data_sorted['timestamp'],
            y=meter_data_sorted['tensao_v'],
            mode='lines',
            name='Tensão (V)',
            line=dict(color='#00A9CE', width=2)
        ))
        
        # Limites PRODIST
        fig.add_hline(y=133, line_dash="dash", line_color="red", 
                     annotation_text="Limite Máx Adequado (133V)")
        fig.add_hline(y=117, line_dash="dash", line_color="red",
                     annotation_text="Limite Mín Adequado (117V)")
        
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title="",
            yaxis_title="Volts",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Linhas pontilhadas indicam limites: Adequado (117-133V), Precário e Crítico.")
    
    with col2:
        st.markdown('<div class="section-title">Curva de Carga & Fator de Potência</div>', unsafe_allow_html=True)
        
        hourly_meter = meter_data.groupby('hora').agg({
            'potencia_kw': 'mean',
            'fator_potencia': 'mean'
        }).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=hourly_meter['hora'],
            y=hourly_meter['potencia_kw'],
            name='Consumo Ativo (kW)',
            marker_color='#FFB300'
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title="Hora",
            yaxis_title="kW",
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "⚡ Motor de Eventos":
    st.markdown('<div class="section-title">⚡ Motor de Regras e Eventos</div>', unsafe_allow_html=True)
    st.markdown("Regras determinísticas aplicadas aos dados brutos para gerar insights operacionais e comerciais.")
    
    if not events_df.empty:
        # Filtros
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tipo_filter = st.multiselect("Tipo", events_df['tipo'].unique())
        with col2:
            sev_filter = st.multiselect("Severidade", events_df['severidade'].unique())
        with col3:
            dest_filter = st.multiselect("Destino", events_df['destino'].unique())
        
        # Aplicar filtros
        filtered = events_df.copy()
        if tipo_filter:
            filtered = filtered[filtered['tipo'].isin(tipo_filter)]
        if sev_filter:
            filtered = filtered[filtered['severidade'].isin(sev_filter)]
        if dest_filter:
            filtered = filtered[filtered['destino'].isin(dest_filter)]
        
        st.markdown("---")
        
        # Tabela de eventos
        st.dataframe(
            filtered[['id_evento', 'id_medidor', 'tipo', 'severidade', 'acao_sugerida', 'destino']],
            use_container_width=True,
            height=400
        )
        
        st.markdown(f"**Total de eventos:** {len(filtered)}")
    else:
        st.info("✅ Nenhum evento detectado no período selecionado")

else:  # GIS / ADMS / SAP
    st.markdown('<div class="section-title">🔧 Integrações Corporativas (Mock)</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🗺️ GIS", "⚡ ADMS", "💼 SAP"])
    
    with tab1:
        st.markdown("### Sistema de Informações Geográficas (GIS)")
        st.markdown("Integração com base cartográfica e ativos da rede.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**Alimentadores Monitorados**")
            for alim in df['alimentador'].unique():
                carga = df[df['alimentador'] == alim]['potencia_kw'].sum() / 1000
                st.markdown(f"- {alim}: {carga:.1f} MW")
        
        with col2:
            st.success("**Status de Comunicação**")
            st.markdown(f"- Medidores Online: {num_meters}")
            st.markdown(f"- Taxa de Comunicação: 99.2%")
    
    with tab2:
        st.markdown("### Advanced Distribution Management System (ADMS)")
        st.markdown("Correlação de eventos de qualidade e interrupções.")
        
        if not events_df.empty:
            criticos = events_df[events_df['severidade'] == 'CRÍTICA']
            if not criticos.empty:
                st.warning(f"⚠️ {len(criticos)} eventos críticos detectados - possível interrupção em andamento")
                
                for _, evt in criticos.head(3).iterrows():
                    st.markdown(f"""
                    <div class="section-card">
                    <strong>{evt['id_medidor']}</strong> - {evt['tipo']}<br>
                    <small>{evt['timestamp']}</small><br>
                    💡 Ação: {evt['acao_sugerida']}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.success("✅ Rede operando normalmente")
    
    with tab3:
        st.markdown("### Sistema SAP (ERP)")
        st.markdown("Geração automática de ordens de serviço a partir de eventos críticos.")
        
        if not events_df.empty:
            os_generated = len(events_df[events_df['severidade'].isin(['CRÍTICA', 'ALTA'])])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("OS Geradas", os_generated)
            col2.metric("Custo Estimado", f"R$ {os_generated * 850:,.2f}")
            col3.metric("Equipes Necessárias", max(1, os_generated // 5))
            
            st.markdown("---")
            st.markdown("**Últimas OS Criadas:**")
            
            for i, (_, evt) in enumerate(events_df[events_df['severidade'].isin(['CRÍTICA', 'ALTA'])].head(5).iterrows(), 1):
                st.markdown(f"""
                <div class="section-card">
                <strong>OS-{20250100 + i}</strong> | {evt['tipo']}<br>
                Medidor: {evt['id_medidor']}<br>
                Prioridade: {evt['severidade']}<br>
                <small>Ação: {evt['acao_sugerida']}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Nenhuma OS gerada automaticamente no período")

# Footer
st.markdown("---")
st.markdown("""
<div class="status-footer">
<strong>Status do Laboratório:</strong> Ambiente de teste (Sandbox) | Dados sintéticos carregados | 
<strong>Versão:</strong> Python 3.10 | Pandas 2.0.1 | Streamlit Mock | 
<strong>Projeto:</strong> P&D ANEEL - CPFL Energia | Tema 3: Smart Meters
</div>
""", unsafe_allow_html=True)
