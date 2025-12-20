import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandapower as pp
import pandapower.networks as pn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import streamlit as st
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# ============================================================================
st.set_page_config(
    page_title="Plataforma Inteligente MMGD",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# FUNÇÕES DE CARREGAMENTO DE DADOS
# ============================================================================

@st.cache_data
def carregar_dados_usinas():
    """Simula dados de usinas MMGD da ANEEL"""
    np.random.seed(42)
    n_usinas = 100
    
    usinas = pd.DataFrame({
        "id": range(1, n_usinas + 1),
        "potencia_kw": np.random.choice([75, 150, 300, 500, 750, 1000, 1500], n_usinas),
        "lat": np.random.uniform(-23.7, -23.4, n_usinas),
        "lon": np.random.uniform(-46.8, -46.5, n_usinas),
        "tipo": np.random.choice(['Solar', 'Eólica', 'Biomassa'], n_usinas, p=[0.92, 0.05, 0.03]),
        "classe": np.random.choice(['Residencial', 'Comercial', 'Industrial'], n_usinas, p=[0.7, 0.2, 0.1]),
        "municipio": np.random.choice(['São Paulo', 'Guarulhos', 'Osasco', 'Santo André'], n_usinas),
        "data_instalacao": pd.date_range(end=datetime.now(), periods=n_usinas, freq='5D')
    })
    
    # Criar geometria para GeoDataFrame
    usinas["geometry"] = usinas.apply(lambda x: Point(x["lon"], x["lat"]), axis=1)
    gdf_usinas = gpd.GeoDataFrame(usinas, geometry="geometry", crs="EPSG:4326")
    
    return gdf_usinas

@st.cache_data
def gerar_dados_clima():
    """Gera dados simulados de clima e geração"""
    horas = list(range(24))
    
    # Irradiância solar (W/m²)
    irradiancia = [0, 0, 0, 0, 0, 50, 150, 300, 500, 700, 850, 950, 
                   1000, 950, 850, 700, 500, 300, 150, 50, 0, 0, 0, 0]
    
    # Temperatura (°C)
    temperatura = [18, 17, 16, 16, 17, 19, 22, 25, 28, 30, 32, 33,
                   34, 33, 32, 30, 28, 25, 23, 21, 20, 19, 18, 18]
    
    clima = pd.DataFrame({
        'hora': horas,
        'irradiancia': irradiancia,
        'temperatura': temperatura
    })
    
    # Calcular geração (modelo simplificado)
    clima['geracao_kw'] = clima['irradiancia'] * 0.15 * (1 - 0.004 * (clima['temperatura'] - 25))
    clima['geracao_kw'] = clima['geracao_kw'].clip(lower=0)
    
    return clima

@st.cache_data
def gerar_dados_ons():
    """Simula dados do ONS - Operador Nacional do Sistema"""
    horas = ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']
    
    dados_ons = pd.DataFrame({
        'hora': horas,
        'carga_mw': [45000, 40000, 48000, 55000, 58000, 56000, 62000, 54000],
        'geracao_convencional_mw': [42000, 38000, 44000, 48000, 50000, 49000, 55000, 50000],
        'geracao_renovavel_mw': [8500, 7200, 8800, 9500, 10200, 9800, 10500, 9200],
        'geracao_mmgd_mw': [150, 100, 450, 1200, 1850, 1650, 850, 200]
    })
    
    return dados_ons

@st.cache_resource
def criar_rede_pandapower():
    """Cria uma rede elétrica usando pandapower"""
    # Criar rede vazia
    net = pp.create_empty_network()
    
    # Criar barramentos
    b1 = pp.create_bus(net, vn_kv=13.8, name="Bus 1")
    b2 = pp.create_bus(net, vn_kv=13.8, name="Bus 2")
    b3 = pp.create_bus(net, vn_kv=13.8, name="Bus 3")
    b4 = pp.create_bus(net, vn_kv=13.8, name="Bus 4")
    b5 = pp.create_bus(net, vn_kv=13.8, name="Bus 5")
    b6 = pp.create_bus(net, vn_kv=13.8, name="Bus 6")
    b7 = pp.create_bus(net, vn_kv=13.8, name="Bus 7")
    b8 = pp.create_bus(net, vn_kv=13.8, name="Bus 8")
    
    # Criar fonte externa (subestação)
    pp.create_ext_grid(net, bus=b1, vm_pu=1.02, name="Grid Connection")
    
    # Criar linhas
    pp.create_line(net, from_bus=b1, to_bus=b2, length_km=2.5, std_type="NAYY 4x150 SE")
    pp.create_line(net, from_bus=b2, to_bus=b3, length_km=1.8, std_type="NAYY 4x150 SE")
    pp.create_line(net, from_bus=b3, to_bus=b4, length_km=2.0, std_type="NAYY 4x150 SE")
    pp.create_line(net, from_bus=b2, to_bus=b5, length_km=1.5, std_type="NAYY 4x150 SE")
    pp.create_line(net, from_bus=b5, to_bus=b6, length_km=2.2, std_type="NAYY 4x150 SE")
    pp.create_line(net, from_bus=b1, to_bus=b7, length_km=3.0, std_type="NAYY 4x150 SE")
    pp.create_line(net, from_bus=b7, to_bus=b8, length_km=1.6, std_type="NAYY 4x150 SE")
    
    # Criar cargas
    pp.create_load(net, bus=b2, p_mw=1.2, q_mvar=0.3, name="Load 1")
    pp.create_load(net, bus=b3, p_mw=0.8, q_mvar=0.2, name="Load 2")
    pp.create_load(net, bus=b4, p_mw=1.5, q_mvar=0.4, name="Load 3")
    pp.create_load(net, bus=b5, p_mw=0.6, q_mvar=0.15, name="Load 4")
    pp.create_load(net, bus=b6, p_mw=1.0, q_mvar=0.25, name="Load 5")
    pp.create_load(net, bus=b7, p_mw=0.9, q_mvar=0.22, name="Load 6")
    pp.create_load(net, bus=b8, p_mw=1.1, q_mvar=0.28, name="Load 7")
    
    # Criar geradores MMGD (Solar/Eólica)
    pp.create_sgen(net, bus=b3, p_mw=0.5, q_mvar=0, name="MMGD Solar 1", type="PV")
    pp.create_sgen(net, bus=b4, p_mw=0.75, q_mvar=0, name="MMGD Solar 2", type="PV")
    pp.create_sgen(net, bus=b6, p_mw=0.3, q_mvar=0, name="MMGD Solar 3", type="PV")
    pp.create_sgen(net, bus=b8, p_mw=1.0, q_mvar=0, name="MMGD Eólica 1", type="WP")
    
    return net

# ============================================================================
# FUNÇÕES DE ANÁLISE
# ============================================================================

def executar_fluxo_potencia(net):
    """Executa o fluxo de potência e adiciona indicadores"""
    try:
        pp.runpp(net)
        
        # Análise de tensão
        net.res_bus["sobretensao"] = net.res_bus.vm_pu > 1.05
        net.res_bus["subtensao"] = net.res_bus.vm_pu < 0.95
        net.res_bus["tensao_ok"] = (net.res_bus.vm_pu >= 0.95) & (net.res_bus.vm_pu <= 1.05)
        
        # Análise de fluxo
        net.res_line["fluxo_reverso"] = net.res_line.p_from_mw < 0
        net.res_line["carregamento"] = abs(net.res_line.loading_percent) / 100
        net.res_line["sobrecarga"] = net.res_line["carregamento"] > 0.8
        
        return True, "Fluxo de potência executado com sucesso!"
    except Exception as e:
        return False, f"Erro ao executar fluxo de potência: {str(e)}"

def calcular_indicadores_rede(net):
    """Calcula indicadores técnicos da rede"""
    indicadores = {}
    
    # Indicadores de tensão
    indicadores['n_sobretensao'] = net.res_bus["sobretensao"].sum()
    indicadores['n_subtensao'] = net.res_bus["subtensao"].sum()
    indicadores['perc_sobretensao'] = (net.res_bus["sobretensao"].sum() / len(net.res_bus)) * 100
    indicadores['perc_subtensao'] = (net.res_bus["subtensao"].sum() / len(net.res_bus)) * 100
    indicadores['tensao_media'] = net.res_bus.vm_pu.mean()
    indicadores['std_tensao'] = net.res_bus.vm_pu.std()
    
    # Indicadores de fluxo
    indicadores['n_fluxo_reverso'] = net.res_line["fluxo_reverso"].sum()
    indicadores['perc_fluxo_reverso'] = (net.res_line["fluxo_reverso"].sum() / len(net.res_line)) * 100
    indicadores['n_sobrecarga'] = net.res_line["sobrecarga"].sum()
    
    # Perdas técnicas
    indicadores['perdas_totais_mw'] = net.res_line.pl_mw.sum()
    indicadores['perdas_totais_mvar'] = net.res_line.ql_mvar.sum()
    
    # Índice de impacto MMGD
    indicadores['indice_impacto'] = (
        0.4 * indicadores['perc_sobretensao'] +
        0.3 * indicadores['perc_fluxo_reverso'] +
        0.3 * indicadores['std_tensao'] * 100
    )
    
    return indicadores

def treinar_modelo_previsao(clima):
    """Treina modelo de previsão de geração"""
    X = clima[["hora", "irradiancia", "temperatura"]]
    y = clima["geracao_kw"]
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Treinar modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Fazer previsões
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Adicionar previsões ao dataframe completo
    clima["geracao_prevista"] = model.predict(X)
    clima["risco_sobretensao"] = clima["geracao_prevista"] > 0.9 * clima["geracao_kw"].max()
    
    return model, mae, r2, clima

# ============================================================================
# FUNÇÕES DE VISUALIZAÇÃO
# ============================================================================

def criar_mapa_usinas(gdf_usinas):
    """Cria mapa interativo das usinas"""
    fig = px.scatter_mapbox(
        gdf_usinas,
        lat=gdf_usinas.geometry.y,
        lon=gdf_usinas.geometry.x,
        size="potencia_kw",
        color="tipo",
        hover_name="id",
        hover_data=["potencia_kw", "classe", "municipio"],
        zoom=10,
        mapbox_style="carto-positron",
        title="Mapa de Usinas MMGD - Região Metropolitana de São Paulo",
        color_discrete_map={'Solar': '#FDB813', 'Eólica': '#0096FF', 'Biomassa': '#228B22'}
    )
    
    fig.update_layout(height=600)
    return fig

def criar_grafico_tensao(net):
    """Cria gráfico de perfil de tensão"""
    df_tensao = net.res_bus.copy()
    df_tensao['barramento'] = [f'B{i+1}' for i in range(len(df_tensao))]
    df_tensao['status'] = df_tensao.apply(
        lambda x: 'Sobretensão' if x['sobretensao'] 
        else ('Subtensão' if x['subtensao'] else 'Normal'), 
        axis=1
    )
    
    fig = px.bar(
        df_tensao,
        x='barramento',
        y='vm_pu',
        color='status',
        title='Perfil de Tensão por Barramento',
        labels={'vm_pu': 'Tensão (p.u.)', 'barramento': 'Barramento'},
        color_discrete_map={'Normal': '#10b981', 'Sobretensão': '#ef4444', 'Subtensão': '#f59e0b'}
    )
    
    # Adicionar linhas de referência
    fig.add_hline(y=1.05, line_dash="dash", line_color="red", annotation_text="Limite Superior")
    fig.add_hline(y=0.95, line_dash="dash", line_color="orange", annotation_text="Limite Inferior")
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray", annotation_text="Nominal")
    
    return fig

def criar_grafico_fluxo(net):
    """Cria gráfico de fluxo de potência"""
    df_fluxo = net.res_line.copy()
    df_fluxo['linha'] = [f'L{i+1}' for i in range(len(df_fluxo))]
    df_fluxo['tipo'] = df_fluxo['fluxo_reverso'].apply(lambda x: 'Reverso' if x else 'Normal')
    
    fig = px.bar(
        df_fluxo,
        x='linha',
        y='p_from_mw',
        color='tipo',
        title='Fluxo de Potência nas Linhas',
        labels={'p_from_mw': 'Potência (MW)', 'linha': 'Linha'},
        color_discrete_map={'Normal': '#3b82f6', 'Reverso': '#f59e0b'}
    )
    
    fig.add_hline(y=0, line_dash="solid", line_color="black")
    
    return fig

def criar_grafico_geracao_ons(dados_ons):
    """Cria gráfico de geração ONS"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dados_ons['hora'], 
        y=dados_ons['carga_mw'],
        name='Carga Total',
        line=dict(color='#3b82f6', width=3),
        mode='lines+markers'
    ))
    
    fig.add_trace(go.Scatter(
        x=dados_ons['hora'], 
        y=dados_ons['geracao_convencional_mw'],
        name='Geração Convencional',
        line=dict(color='#6366f1', width=2),
        fill='tozeroy'
    ))
    
    fig.add_trace(go.Scatter(
        x=dados_ons['hora'], 
        y=dados_ons['geracao_renovavel_mw'],
        name='Geração Renovável',
        line=dict(color='#10b981', width=2),
        fill='tozeroy'
    ))
    
    fig.add_trace(go.Scatter(
        x=dados_ons['hora'], 
        y=dados_ons['geracao_mmgd_mw'],
        name='MMGD',
        line=dict(color='#f59e0b', width=2),
        mode='lines+markers'
    ))
    
    fig.update_layout(
        title='Curva de Carga ONS - Sistema Interligado Nacional',
        xaxis_title='Hora',
        yaxis_title='Potência (MW)',
        hovermode='x unified',
        height=400
    )
    
    return fig

def criar_grafico_previsao(clima):
    """Cria gráfico de previsão de geração"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=clima['hora'],
        y=clima['geracao_kw'],
        name='Geração Real',
        line=dict(color='#3b82f6', width=2),
        mode='lines+markers'
    ))
    
    fig.add_trace(go.Scatter(
        x=clima['hora'],
        y=clima['geracao_prevista'],
        name='Geração Prevista',
        line=dict(color='#f59e0b', width=2, dash='dash'),
        mode='lines+markers'
    ))
    
    # Destacar períodos de risco
    risco_horas = clima[clima['risco_sobretensao']]['hora']
    if len(risco_horas) > 0:
        fig.add_trace(go.Scatter(
            x=risco_horas,
            y=clima[clima['risco_sobretensao']]['geracao_kw'],
            mode='markers',
            name='Risco de Sobretensão',
            marker=dict(color='red', size=12, symbol='x')
        ))
    
    fig.update_layout(
        title='Previsão de Geração Solar - Machine Learning',
        xaxis_title='Hora do Dia',
        yaxis_title='Geração (kW)',
        hovermode='x unified',
        height=400
    )
    
    return fig

# ============================================================================
# INTERFACE STREAMLIT
# ============================================================================

def main():
    """Função principal da aplicação"""
    # Título e descrição
    st.title("⚡ Plataforma Inteligente MMGD")
    st.markdown("### Integração ONS/ANEEL - Análise de Micro e Minigeração Distribuída")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        aba = st.radio(
            "Selecione a análise:",
            ["📊 Visão Geral", "🔌 Análise de Rede", "🗺️ Mapa de Usinas", "🤖 Machine Learning"]
        )
        
        st.markdown("---")
        st.markdown("### 📈 Estatísticas Gerais")
        st.metric("Total de Usinas", "1.247.589")
        st.metric("Potência Instalada", "18,5 GW")
        st.metric("Economia Anual", "24,6 TWh")
    
    # Carregar dados
    gdf_usinas = carregar_dados_usinas()
    dados_ons = gerar_dados_ons()
    clima = gerar_dados_clima()
    net = criar_rede_pandapower()
    
    # Executar fluxo de potência
    sucesso, mensagem = executar_fluxo_potencia(net)
    
    if not sucesso:
        st.error(mensagem)
        return
    
    indicadores = calcular_indicadores_rede(net)
    
    # ========================================================================
    # ABA: VISÃO GERAL
    # ========================================================================
    if aba == "📊 Visão Geral":
        st.header("📊 Visão Geral do Sistema")
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Índice de Impacto MMGD", 
                f"{indicadores['indice_impacto']:.2f}",
                delta="-2.3% vs mês anterior",
                delta_color="inverse"
            )
        
        with col2:
            st.metric(
                "Barramentos OK", 
                f"{len(net.res_bus) - indicadores['n_sobretensao'] - indicadores['n_subtensao']}/{len(net.res_bus)}",
                delta=f"{100 - indicadores['perc_sobretensao'] - indicadores['perc_subtensao']:.1f}%"
            )
        
        with col3:
            municipios_selecionados = st.multiselect(
                "Município:",
                options=gdf_usinas['municipio'].unique(),
                default=gdf_usinas['municipio'].unique(),
                key="municipios_geral"
            )
        
        # Definir padrões para tipos e classes caso não estejam filtrados nesta aba
        # para evitar erros na filtragem abaixo
        tipos_selecionados = gdf_usinas['tipo'].unique()
        classes_selecionadas = gdf_usinas['classe'].unique()
        
        # Filtrar dados
        gdf_filtrado = gdf_usinas[
            (gdf_usinas['tipo'].isin(tipos_selecionados)) &
            (gdf_usinas['classe'].isin(classes_selecionadas)) &
            (gdf_usinas['municipio'].isin(municipios_selecionados))
        ]
        
        # Exibir estatísticas filtradas (linha 2 de métricas)
        # Atenção: Adicionada lógica para evitar erro de colunas vazias
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Usinas Filtradas", len(gdf_filtrado))
        
        with col2:
            st.metric("Potência Total", f"{gdf_filtrado['potencia_kw'].sum()/1000:.2f} MW")
        
        with col3:
            st.metric("Potência Média", f"{gdf_filtrado['potencia_kw'].mean():.0f} kW")
        
        with col4:
            st.metric("Potência Máxima", f"{gdf_filtrado['potencia_kw'].max()} kW")
        
        st.markdown("---")
        
        # Gráficos principais
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(criar_grafico_geracao_ons(dados_ons), use_container_width=True)
        
        with col2:
            # Distribuição por tipo
            dist_tipo = gdf_usinas.groupby('tipo')['potencia_kw'].sum().reset_index()
            fig_pie = px.pie(
                dist_tipo, 
                values='potencia_kw', 
                names='tipo',
                title='Distribuição de Potência por Fonte',
                color_discrete_map={'Solar': '#FDB813', 'Eólica': '#0096FF', 'Biomassa': '#228B22'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Estatísticas detalhadas
        st.markdown("### 📋 Estatísticas Detalhadas")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🔌 Usinas por Classe")
            dist_classe = gdf_usinas['classe'].value_counts()
            for classe, count in dist_classe.items():
                st.write(f"**{classe}:** {count} usinas")
        
        with col2:
            st.markdown("#### 📍 Usinas por Município")
            dist_municipio = gdf_usinas['municipio'].value_counts().head(5)
            for municipio, count in dist_municipio.items():
                st.write(f"**{municipio}:** {count} usinas")
        
        with col3:
            st.markdown("#### ⚡ Potência Instalada")
            potencia_total = gdf_usinas['potencia_kw'].sum()
            potencia_media = gdf_usinas['potencia_kw'].mean()
            st.write(f"**Total:** {potencia_total/1000:.2f} MW")
            st.write(f"**Média:** {potencia_media:.2f} kW")
            st.write(f"**Máxima:** {gdf_usinas['potencia_kw'].max()} kW")
    
    # ========================================================================
    # ABA: ANÁLISE DE REDE
    # ========================================================================
    elif aba == "🔌 Análise de Rede":
        st.header("🔌 Análise Técnica da Rede Elétrica")
        
        # Alertas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if indicadores['n_sobretensao'] > 0:
                st.error(f"⚠️ {indicadores['n_sobretensao']} barramentos em sobretensão")
            else:
                st.success("✅ Sem sobretensão")
        
        with col2:
            if indicadores['n_subtensao'] > 0:
                st.warning(f"⚠️ {indicadores['n_subtensao']} barramentos em subtensão")
            else:
                st.success("✅ Sem subtensão")
        
        with col3:
            if indicadores['n_fluxo_reverso'] > 0:
                st.info(f"ℹ️ {indicadores['n_fluxo_reverso']} linhas com fluxo reverso")
            else:
                st.success("✅ Sem fluxo reverso")
        
        with col4:
            if indicadores['n_sobrecarga'] > 0:
                st.error(f"⚠️ {indicadores['n_sobrecarga']} linhas em sobrecarga")
            else:
                st.success("✅ Sem sobrecarga")
        
        st.markdown("---")
        
        # Gráficos de análise
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(criar_grafico_tensao(net), use_container_width=True)
        
        with col2:
            st.plotly_chart(criar_grafico_fluxo(net), use_container_width=True)
        
        # Tabela detalhada de barramentos
        st.markdown("### 📊 Análise Detalhada dos Barramentos")
        
        df_display = net.res_bus.copy()
        df_display['barramento'] = [f'B{i+1}' for i in range(len(df_display))]
        df_display['status'] = df_display.apply(
            lambda x: '🔴 Sobretensão' if x['sobretensao'] 
            else ('🟠 Subtensão' if x['subtensao'] else '🟢 Normal'), 
            axis=1
        )
        
        st.dataframe(
            df_display[['barramento', 'vm_pu', 'va_degree', 'p_mw', 'q_mvar', 'status']],
            hide_index=True,
            use_container_width=True
        )
        
        # Tabela detalhada de linhas
        st.markdown("### 📊 Análise Detalhada das Linhas")
        
        df_linhas = net.res_line.copy()
        df_linhas['linha'] = [f'L{i+1}' for i in range(len(df_linhas))]
        df_linhas['fluxo_status'] = df_linhas['fluxo_reverso'].apply(
            lambda x: '🔄 Reverso' if x else '➡️ Normal'
        )
        df_linhas['carga_status'] = df_linhas['sobrecarga'].apply(
            lambda x: '⚠️ Sobrecarga' if x else '✅ OK'
        )
        
        st.dataframe(
            df_linhas[['linha', 'p_from_mw', 'loading_percent', 'pl_mw', 'fluxo_status', 'carga_status']],
            hide_index=True,
            use_container_width=True
        )
    
    # ========================================================================
    # ABA: MAPA DE USINAS
    # ========================================================================
    elif aba == "🗺️ Mapa de Usinas":
        st.header("🗺️ Mapeamento de Usinas MMGD")
        
        # Filtros
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tipos_selecionados = st.multiselect(
                "Tipo de Fonte:",
                options=gdf_usinas['tipo'].unique(),
                default=gdf_usinas['tipo'].unique(),
                key="tipo_mapa"
            )
        
        with col2:
            classes_selecionadas = st.multiselect(
                "Classe de Consumo:",
                options=gdf_usinas['classe'].unique(),
                default=gdf_usinas['classe'].unique(),
                key="classe_mapa"
            )
        
        with col3:
            municipios_selecionados = st.multiselect(
                "Município:",
                options=gdf_usinas['municipio'].unique(),
                default=gdf_usinas['municipio'].unique(),
                key="municipio_mapa"
            )

        # Filtrar dados para o mapa
        gdf_filtrado_mapa = gdf_usinas[
            (gdf_usinas['tipo'].isin(tipos_selecionados)) &
            (gdf_usinas['classe'].isin(classes_selecionadas)) &
            (gdf_usinas['municipio'].isin(municipios_selecionados))
        ]

        st.markdown("---")

        col1, col2 = st.columns([3, 1])

        with col1:
            # Mapa interativo
            if len(gdf_filtrado_mapa) > 0:
                st.plotly_chart(criar_mapa_usinas(gdf_filtrado_mapa), use_container_width=True)
            else:
                st.warning("Nenhuma usina encontrada com os filtros selecionados.")

        with col2:
            st.markdown("### Resumo")
            st.metric("Usinas Listadas", len(gdf_filtrado_mapa))
            st.metric("Potência Total", f"{gdf_filtrado_mapa['potencia_kw'].sum()/1000:.2f} MW")

        # Tabela de dados do mapa
        with st.expander("Ver dados brutos"):
            st.dataframe(
                gdf_filtrado_mapa[['id', 'tipo', 'classe', 'municipio', 'potencia_kw']],
                hide_index=True,
                use_container_width=True
            )

    # ========================================================================
    # ABA: MACHINE LEARNING
    # ========================================================================
    elif aba == "🤖 Machine Learning":
        st.header("🤖 Previsão de Geração com Machine Learning")
        
        # Treinar modelo
        with st.spinner("Treinando modelo Random Forest..."):
            model, mae, r2, clima_previsto = treinar_modelo_previsao(clima)
        
        # Métricas do modelo
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Algoritmo", "Random Forest")
        
        with col2:
            st.metric("MAE (kW)", f"{mae:.2f}")
        
        with col3:
            st.metric("R² Score", f"{r2:.4f}")
        
        with col4:
            n_risco = clima_previsto['risco_sobretensao'].sum()
            st.metric("Horas em Risco", n_risco)
        
        st.markdown("---")
        
        # Gráfico de previsão
        st.plotly_chart(criar_grafico_previsao(clima_previsto), use_container_width=True)
        
        # Dados climáticos
        col1, col2 = st.columns(2)
        
        with col1:
            fig_irrad = px.line(
                clima_previsto,
                x='hora',
                y='irradiancia',
                title='Irradiância Solar ao Longo do Dia',
                labels={'irradiancia': 'Irradiância (W/m²)', 'hora': 'Hora do Dia'}
            )
            st.plotly_chart(fig_irrad, use_container_width=True)
        
        with col2:
            fig_temp = px.line(
                clima_previsto,
                x='hora',
                y='temperatura',
                title='Temperatura ao Longo do Dia',
                labels={'temperatura': 'Temperatura (°C)', 'hora': 'Hora do Dia'},
                line_shape='spline'
            )
            st.plotly_chart(fig_temp, use_container_width=True)
        
        # Tabela de dados
        st.markdown("### 📊 Dados Climáticos e Previsões")
        
        df_display_ml = clima_previsto.copy()
        df_display_ml['risco'] = df_display_ml['risco_sobretensao'].apply(
            lambda x: '⚠️ Alto Risco' if x else '✅ Normal'
        )
        
        st.dataframe(
            df_display_ml[['hora', 'irradiancia', 'temperatura', 'geracao_kw', 'geracao_prevista', 'risco']],
            hide_index=True,
            use_container_width=True
        )
        
        # Feature Importance
        st.markdown("### 🎯 Importância das Features")
        
        feature_importance = pd.DataFrame({
            'Feature': ['hora', 'irradiancia', 'temperatura'],
            'Importância': model.feature_importances_
        }).sort_values('Importância', ascending=False)
        
        fig_importance = px.bar(
            feature_importance,
            x='Importância',
            y='Feature',
            orientation='h',
            title='Importância das Variáveis no Modelo'
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)

if __name__ == "__main__":
    main()
