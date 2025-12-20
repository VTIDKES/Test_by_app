import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import plotly.express as px
import plotly.graph_objects as go
import pandapower as pp
import pandapower.networks as pn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import streamlit as st
from datetime import datetime, timedelta
import warnings
import requests
import json

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# ============================================================================
st.set_page_config(
    page_title="Plataforma Inteligente MMGD - RN",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# FUNÇÕES DE CARREGAMENTO DE DADOS (AGORA COM API REAL)
# ============================================================================

@st.cache_data(ttl=3600) # Cache por 1 hora para não sobrecarregar a API
def carregar_dados_usinas():
    """
    Busca dados reais de Geração Distribuída da API da ANEEL
    Filtrando apenas pelo estado do Rio Grande do Norte (RN).
    """
    # ID do recurso "Empreendimentos de Geração Distribuída" no CKAN da ANEEL
    # Nota: Esse ID pode mudar periodicamente. Se falhar, verifique no site dadosabertos.aneel.gov.br
    resource_id = "b1bd71e7-d0ad-4214-9053-cbd58e9564a7"
    
    url = "https://dadosabertos.aneel.gov.br/api/3/action/datastore_search"
    
    # Limite de registros (ajuste conforme necessário, o RN tem muitos registros)
    limit = 1000
    
    params = {
        "resource_id": resource_id,
        "filters": '{"SigUF": "RN"}', # Filtro JSON para Rio Grande do Norte
        "limit": limit
    }

    try:
        with st.spinner('Conectando à API da ANEEL (Dados do RN)...'):
            response = requests.get(url, params=params)
            response.raise_for_status() # Levanta erro se a requisição falhar
            
            dados = response.json()
            
            if not dados['success']:
                st.error("Erro na resposta da API da ANEEL")
                return pd.DataFrame() # Retorna vazio em caso de erro lógico
            
            records = dados['result']['records']
            df = pd.DataFrame(records)

            if df.empty:
                st.warning("Nenhum dado encontrado para o RN. Verifique a API.")
                return pd.DataFrame()

            # --- TRATAMENTO DOS DADOS ---
            
            # 1. Mapear colunas da ANEEL para nomes internos
            # Os nomes das colunas da ANEEL podem variar, mas geralmente seguem esse padrão:
            df_tratado = pd.DataFrame()
            df_tratado['id'] = df.get('CodEmpreendimento', df.index)
            df_tratado['municipio'] = df.get('NomMunicipio', 'Desconhecido')
            df_tratado['classe'] = df.get('DscClasseConsumo', 'Não Informado')
            df_tratado['tipo'] = df.get('DscFonteGeracao', 'Outra')
            df_tratado['data_instalacao'] = pd.to_datetime(df.get('DthAtualizaCadastralEmpreend', datetime.now()), errors='coerce')
            
            # 2. Tratamento Numérico (Potência e Coordenadas costumam vir como string com vírgula)
            def limpar_numero(val):
                if isinstance(val, str):
                    return float(val.replace(',', '.'))
                return float(val)

            df_tratado['potencia_kw'] = df.get('MdaPotenciaInstaladaKW', 0).apply(limpar_numero)
            
            # Tentar pegar latitude/longitude. A ANEEL nem sempre preenche bem isso.
            # Se não tiver, vamos simular coordenadas DENTRO do RN para não quebrar o mapa
            if 'NumCoordNEmpreendimento' in df.columns:
                df_tratado['lat'] = df['NumCoordNEmpreendimento'].apply(limpar_numero)
                df_tratado['lon'] = df['NumCoordEEmpreendimento'].apply(limpar_numero)
            else:
                # Fallback: Simular lat/lon espalhados pelo RN se a API não trouxer
                np.random.seed(42)
                n_rows = len(df)
                # Box aproximado do RN
                df_tratado['lat'] = np.random.uniform(-6.5, -5.0, n_rows)
                df_tratado['lon'] = np.random.uniform(-37.5, -35.0, n_rows)

            # Remover coordenadas zeradas ou inválidas
            df_tratado = df_tratado[(df_tratado['lat'] != 0) & (df_tratado['lon'] != 0)]
            df_tratado.dropna(subset=['lat', 'lon'], inplace=True)

            # Criar geometria
            df_tratado["geometry"] = df_tratado.apply(lambda x: Point(x["lon"], x["lat"]), axis=1)
            gdf = gpd.GeoDataFrame(df_tratado, geometry="geometry", crs="EPSG:4326")
            
            return gdf

    except Exception as e:
        st.error(f"Falha ao conectar na API: {str(e)}")
        # Retorna dados simulados do RN como fallback para a apresentação não parar
        return gerar_dados_simulados_rn_fallback()

def gerar_dados_simulados_rn_fallback():
    """Função de backup caso a API esteja fora do ar"""
    np.random.seed(42)
    n_usinas = 100
    usinas = pd.DataFrame({
        "id": range(1, n_usinas + 1),
        "potencia_kw": np.random.choice([3, 5, 10, 30, 75], n_usinas),
        # Coordenadas aproximadas do RN (Natal, Mossoró, etc)
        "lat": np.random.uniform(-6.0, -5.7, n_usinas), 
        "lon": np.random.uniform(-35.3, -35.1, n_usinas),
        "tipo": np.random.choice(['Radiação Solar', 'Eólica', 'Biomassa'], n_usinas, p=[0.90, 0.08, 0.02]),
        "classe": np.random.choice(['Residencial', 'Comercial', 'Industrial'], n_usinas),
        "municipio": np.random.choice(['Natal', 'Mossoró', 'Parnamirim', 'Caicó'], n_usinas),
        "data_instalacao": pd.date_range(end=datetime.now(), periods=n_usinas, freq='5D')
    })
    usinas["geometry"] = usinas.apply(lambda x: Point(x["lon"], x["lat"]), axis=1)
    return gpd.GeoDataFrame(usinas, geometry="geometry", crs="EPSG:4326")

@st.cache_data
def gerar_dados_clima():
    """Gera dados simulados de clima do Nordeste"""
    horas = list(range(24))
    # Irradiância mais forte (Nordeste)
    irradiancia = [0, 0, 0, 0, 0, 100, 300, 550, 800, 950, 1000, 1050, 
                   1000, 900, 750, 550, 300, 100, 0, 0, 0, 0, 0, 0]
    temperatura = [24, 23, 23, 22, 23, 25, 27, 29, 31, 32, 33, 34,
                   33, 32, 31, 30, 28, 27, 26, 25, 25, 24, 24, 24]
    
    clima = pd.DataFrame({'hora': horas, 'irradiancia': irradiancia, 'temperatura': temperatura})
    clima['geracao_kw'] = clima['irradiancia'] * 0.15 * (1 - 0.004 * (clima['temperatura'] - 25))
    clima['geracao_kw'] = clima['geracao_kw'].clip(lower=0)
    return clima

@st.cache_data
def gerar_dados_ons_nordeste():
    """Simula dados do ONS para o Subsistema Nordeste"""
    horas = ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']
    dados_ons = pd.DataFrame({
        'hora': horas,
        'carga_mw': [10000, 9500, 10500, 12000, 13000, 12500, 14000, 12000],
        'geracao_convencional_mw': [2000, 1500, 2000, 1000, 1000, 1000, 3000, 2500],
        'geracao_renovavel_mw': [7800, 7800, 8200, 10000, 11000, 10500, 10000, 9000], # Forte eólica/solar
        'geracao_mmgd_mw': [0, 0, 300, 1000, 1200, 1000, 100, 0]
    })
    return dados_ons

@st.cache_resource
def criar_rede_pandapower():
    """Cria uma rede elétrica representativa"""
    net = pp.create_empty_network()
    # Rede simplificada para demonstração de fluxo
    b1 = pp.create_bus(net, vn_kv=13.8, name="Subestação RN")
    b2 = pp.create_bus(net, vn_kv=13.8, name="Alimentador 1")
    b3 = pp.create_bus(net, vn_kv=13.8, name="Carga/GD 1")
    b4 = pp.create_bus(net, vn_kv=13.8, name="Carga/GD 2")
    
    pp.create_ext_grid(net, bus=b1, vm_pu=1.02)
    pp.create_line(net, b1, b2, length_km=2.0, std_type="NAYY 4x150 SE")
    pp.create_line(net, b2, b3, length_km=1.5, std_type="NAYY 4x150 SE")
    pp.create_line(net, b3, b4, length_km=1.0, std_type="NAYY 4x150 SE")
    
    pp.create_load(net, bus=b3, p_mw=0.5, q_mvar=0.1)
    pp.create_load(net, bus=b4, p_mw=0.8, q_mvar=0.2)
    
    # GDs
    pp.create_sgen(net, bus=b3, p_mw=0.4, q_mvar=0, name="GD Solar RN 1", type="PV")
    pp.create_sgen(net, bus=b4, p_mw=0.6, q_mvar=0, name="GD Solar RN 2", type="PV")
    
    return net

# ============================================================================
# FUNÇÕES DE ANÁLISE (FLUXO DE POTÊNCIA)
# ============================================================================

def executar_fluxo_potencia(net):
    try:
        pp.runpp(net)
        net.res_bus["sobretensao"] = net.res_bus.vm_pu > 1.05
        net.res_bus["subtensao"] = net.res_bus.vm_pu < 0.95
        net.res_line["fluxo_reverso"] = net.res_line.p_from_mw < 0
        net.res_line["carregamento"] = abs(net.res_line.loading_percent) / 100
        net.res_line["sobrecarga"] = net.res_line["carregamento"] > 0.8
        return True, "Sucesso"
    except Exception as e:
        return False, str(e)

def calcular_indicadores_rede(net):
    indicadores = {}
    indicadores['n_sobretensao'] = net.res_bus["sobretensao"].sum()
    indicadores['n_subtensao'] = net.res_bus["subtensao"].sum()
    indicadores['n_fluxo_reverso'] = net.res_line["fluxo_reverso"].sum()
    indicadores['perdas_totais_mw'] = net.res_line.pl_mw.sum()
    indicadores['indice_impacto'] = (net.res_bus["sobretensao"].sum() * 0.5 + net.res_line["fluxo_reverso"].sum() * 0.5)
    return indicadores

def treinar_modelo_previsao(clima):
    X = clima[["hora", "irradiancia", "temperatura"]]
    y = clima["geracao_kw"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    clima["geracao_prevista"] = model.predict(X)
    clima["risco_sobretensao"] = clima["geracao_prevista"] > 0.9 * clima["geracao_kw"].max()
    return model, mae, r2, clima

# ============================================================================
# VISUALIZAÇÃO
# ============================================================================

def criar_mapa_usinas(gdf_usinas):
    # Ajustar centro do mapa para o RN
    lat_center = -5.8
    lon_center = -36.5
    
    fig = px.scatter_mapbox(
        gdf_usinas,
        lat=gdf_usinas.geometry.y,
        lon=gdf_usinas.geometry.x,
        size="potencia_kw",
        color="tipo",
        hover_name="municipio",
        hover_data=["potencia_kw", "classe", "id"],
        zoom=7,
        center={"lat": lat_center, "lon": lon_center},
        mapbox_style="carto-positron",
        title="Mapa de Usinas - Rio Grande do Norte (Dados ANEEL)",
        color_discrete_map={
            'Radiação Solar': '#FDB813', 
            'Cinética do Vento': '#0096FF', 
            'Biomassa': '#228B22',
            'Outra': '#808080'
        }
    )
    fig.update_layout(height=600)
    return fig

# ============================================================================
# MAIN
# ============================================================================

def main():
    st.title("⚡ Plataforma Inteligente MMGD - Rio Grande do Norte")
    st.markdown("### Integração de Dados Reais ANEEL (API Dados Abertos)")
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/b/be/Flag_of_Rio_Grande_do_Norte.svg", width=100)
        st.header("Configurações RN")
        aba = st.radio("Módulos:", ["📊 Visão Geral", "🔌 Análise de Rede", "🗺️ Mapa do RN", "🤖 Machine Learning"])
        st.info("Conectado à API: dadosabertos.aneel.gov.br")

    # Carregamento
    gdf_usinas = carregar_dados_usinas()
    dados_ons = gerar_dados_ons_nordeste()
    clima = gerar_dados_clima()
    net = criar_rede_pandapower()
    executar_fluxo_potencia(net)
    indicadores = calcular_indicadores_rede(net)

    # Verifica se carregou dados
    if gdf_usinas.empty:
        st.error("Não foi possível carregar os dados. Verifique a conexão com a API.")
        return

    # ABA: VISÃO GERAL
    if aba == "📊 Visão Geral":
        st.header("📊 Panorama da Geração Distribuída no RN")
        
        total_potencia = gdf_usinas['potencia_kw'].sum() / 1000 # MW
        total_usinas = len(gdf_usinas)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Usinas (RN)", f"{total_usinas}", "Atualizado API")
        col2.metric("Potência Instalada", f"{total_potencia:.2f} MW", "Rio Grande do Norte")
        col3.metric("Municípios Atendidos", f"{gdf_usinas['municipio'].nunique()}", "Cobertura Estadual")

        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            # Gráfico ONS Nordeste
            fig_ons = go.Figure()
            fig_ons.add_trace(go.Scatter(x=dados_ons['hora'], y=dados_ons['carga_mw'], name='Carga NE'))
            fig_ons.add_trace(go.Scatter(x=dados_ons['hora'], y=dados_ons['geracao_renovavel_mw'], name='Renováveis NE'))
            fig_ons.update_layout(title="Curva de Carga x Geração (Subsistema Nordeste)")
            st.plotly_chart(fig_ons, use_container_width=True)
            
        with col2:
            # Pizza por Tipo
            fig_pie = px.pie(gdf_usinas, names='tipo', values='potencia_kw', title='Matriz GD no RN')
            st.plotly_chart(fig_pie, use_container_width=True)

    # ABA: ANÁLISE DE REDE (Pandapower - Simulação Técnica)
    elif aba == "🔌 Análise de Rede":
        st.header("🔌 Análise de Fluxo de Potência (Simulação)")
        st.info("Nota: A topologia exata da rede da COSERN/Neoenergia não é pública. Esta análise utiliza um alimentador modelo representativo parametrizado com os dados reais de geração.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Níveis de Tensão Críticos", indicadores['n_sobretensao'] + indicadores['n_subtensao'])
        with col2:
            st.metric("Fluxo Reverso Detectado", f"{indicadores['n_fluxo_reverso']} linhas")
            
        # Perfil de Tensão
        df_tensao = net.res_bus.copy()
        df_tensao['barramento'] = df_tensao.index
        fig_v = px.bar(df_tensao, x='barramento', y='vm_pu', color='vm_pu', 
                      title="Perfil de Tensão nos Barramentos (p.u.)", range_y=[0.9, 1.1])
        fig_v.add_hline(y=1.05, line_dash="dash", line_color="red")
        fig_v.add_hline(y=0.95, line_dash="dash", line_color="orange")
        st.plotly_chart(fig_v, use_container_width=True)

    # ABA: MAPA
    elif aba == "🗺️ Mapa do RN":
        st.header("🗺️ Geolocalização das Usinas - Rio Grande do Norte")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            cidades = st.multiselect("Filtrar Município:", gdf_usinas['municipio'].unique())
            fontes = st.multiselect("Filtrar Fonte:", gdf_usinas['tipo'].unique())
        
        # Filtragem dinâmica
        gdf_filtered = gdf_usinas.copy()
        if cidades:
            gdf_filtered = gdf_filtered[gdf_filtered['municipio'].isin(cidades)]
        if fontes:
            gdf_filtered = gdf_filtered[gdf_filtered['tipo'].isin(fontes)]
            
        st.plotly_chart(criar_mapa_usinas(gdf_filtered), use_container_width=True)
        
        with st.expander("Ver Tabela de Dados (Extraído da ANEEL)"):
            st.dataframe(gdf_filtered[['id', 'municipio', 'tipo', 'classe', 'potencia_kw', 'data_instalacao']])

    # ABA: MACHINE LEARNING
    elif aba == "🤖 Machine Learning":
        st.header("🤖 Previsão de Geração Solar (Modelo RN)")
        model, mae, r2, clima_prev = treinar_modelo_previsao(clima)
        
        c1, c2 = st.columns(2)
        c1.metric("Precisão do Modelo (R²)", f"{r2:.2%}")
        c2.metric("Erro Médio Absoluto", f"{mae:.2f} kW")
        
        fig_pred = px.line(clima_prev, x='hora', y=['geracao_kw', 'geracao_prevista'], 
                          title="Geração Real vs Prevista (Random Forest)")
        st.plotly_chart(fig_pred, use_container_width=True)

if __name__ == "__main__":
    main()

