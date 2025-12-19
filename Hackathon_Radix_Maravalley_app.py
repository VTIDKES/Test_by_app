"""
Sistema Integrado de Gestão Inteligente de Micro e Minigeração Distribuída (MMGD)
Autor: Sistema aprimorado
Data: 2025

Funcionalidades:
- Modelagem de rede elétrica com PandaPower
- Previsão de geração solar com Machine Learning
- Análise de risco e violações de tensão
- Visualização geoespacial
- Interface web interativa com Streamlit
"""

import pandapower as pp
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import geopandas as gpd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ========== CLASSE PRINCIPAL: GERENCIADOR DE REDE ==========
class GridManager:
    """Gerencia a rede elétrica e suas operações"""
    
    def __init__(self):
        self.net = None
        self.forecast_model = None
        self.history = []
        
    def create_network(self):
        """Cria a topologia da rede elétrica"""
        self.net = pp.create_empty_network()
        
        # Criação das barras
        b1 = pp.create_bus(self.net, vn_kv=13.8, name="Subestacao", type="b")
        b2 = pp.create_bus(self.net, vn_kv=13.8, name="Carga_Industrial", type="b")
        b3 = pp.create_bus(self.net, vn_kv=13.8, name="MMGD_Solar", type="b")
        
        # Criação das linhas de transmissão
        pp.create_line_from_parameters(
            self.net, b1, b2, length_km=2.0,
            r_ohm_per_km=0.4, x_ohm_per_km=0.3,
            c_nf_per_km=210, max_i_ka=0.3,
            name="Linha_SE_Industrial"
        )
        pp.create_line_from_parameters(
            self.net, b2, b3, length_km=1.0,
            r_ohm_per_km=0.3, x_ohm_per_km=0.25,
            c_nf_per_km=180, max_i_ka=0.25,
            name="Linha_Industrial_Solar"
        )
        
        # Fonte externa (subestação)
        pp.create_ext_grid(self.net, b1, vm_pu=1.0, name="Rede_Externa")
        
        # Carga industrial
        pp.create_load(self.net, b2, p_mw=1.2, q_mvar=0.4, name="Carga_Industrial")
        
        # Gerador MMGD (inicialmente sem geração)
        pp.create_sgen(self.net, b3, p_mw=0.0, q_mvar=0.0, name="Solar_PV")
        
        return self.net
    
    def run_power_flow(self):
        """Executa o fluxo de potência"""
        try:
            pp.runpp(self.net, algorithm='nr', calculate_voltage_angles=True)
            return True, "Fluxo de potência convergiu"
        except Exception as e:
            return False, f"Erro no fluxo de potência: {str(e)}"
    
    def update_generation(self, power_mw, power_factor=0.95):
        """Atualiza a geração do sistema MMGD"""
        if self.net is not None and len(self.net.sgen) > 0:
            self.net.sgen.at[0, "p_mw"] = power_mw
            # Calcula potência reativa baseada no fator de potência
            q_mvar = power_mw * np.tan(np.arccos(power_factor))
            self.net.sgen.at[0, "q_mvar"] = q_mvar
    
    def update_load(self, bus_idx, p_mw, q_mvar):
        """Atualiza valores de carga"""
        if self.net is not None and len(self.net.load) > 0:
            self.net.load.at[bus_idx, "p_mw"] = p_mw
            self.net.load.at[bus_idx, "q_mvar"] = q_mvar
    
    def get_results(self):
        """Retorna resultados consolidados"""
        if self.net is None:
            return None
        
        results = {
            'buses': self.net.res_bus.copy(),
            'lines': self.net.res_line.copy(),
            'generation': self.net.res_sgen.copy() if len(self.net.res_sgen) > 0 else None,
            'loads': self.net.res_load.copy()
        }
        return results


# ========== CLASSE: PREVISÃO DE GERAÇÃO ==========
class GenerationForecast:
    """Gerencia previsão de geração solar usando ML"""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.is_trained = False
        self.feature_importance = None
        self.metrics = {}
    
    def generate_synthetic_data(self, n_samples=1000):
        """Gera dados sintéticos para treinamento"""
        np.random.seed(42)
        
        # Irradiância solar (W/m²) - distribuição realista
        irradiancia = np.random.gamma(shape=4, scale=200, size=n_samples)
        irradiancia = np.clip(irradiancia, 0, 1000)
        
        # Temperatura (°C)
        temperatura = np.random.normal(28, 5, n_samples)
        temperatura = np.clip(temperatura, 15, 40)
        
        # Hora do dia (0-23)
        hora = np.random.randint(0, 24, n_samples)
        
        # Nebulosidade (0-100%)
        nebulosidade = np.random.beta(2, 5, n_samples) * 100
        
        # Cálculo da potência (modelo simplificado)
        potencia_base = (irradiancia / 1000) * 1.5  # 1.5 MW de capacidade
        fator_temp = 1 - (temperatura - 25) * 0.004  # Perda por temperatura
        fator_nuvem = 1 - (nebulosidade / 100) * 0.7
        fator_hora = np.where((hora >= 6) & (hora <= 18), 1.0, 0.1)
        
        potencia_mw = potencia_base * fator_temp * fator_nuvem * fator_hora
        potencia_mw = np.clip(potencia_mw, 0, 1.5)
        
        # Adiciona ruído
        potencia_mw += np.random.normal(0, 0.02, n_samples)
        potencia_mw = np.clip(potencia_mw, 0, 1.5)
        
        df = pd.DataFrame({
            'irradiancia': irradiancia,
            'temperatura': temperatura,
            'hora': hora,
            'nebulosidade': nebulosidade,
            'potencia_mw': potencia_mw
        })
        
        return df
    
    def train(self, df=None):
        """Treina o modelo de previsão"""
        if df is None:
            df = self.generate_synthetic_data()
        
        # Features e target
        X = df[['irradiancia', 'temperatura', 'hora', 'nebulosidade']]
        y = df['potencia_mw']
        
        # Divisão treino/teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Treinamento
        self.model.fit(X_train, y_train)
        
        # Avaliação
        y_pred = self.model.predict(X_test)
        self.metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(np.mean((y_test - y_pred)**2))
        }
        
        # Importância das features
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.is_trained = True
        return self.metrics
    
    def predict(self, irradiancia, temperatura, hora, nebulosidade):
        """Faz previsão de geração"""
        if not self.is_trained:
            raise ValueError("Modelo não treinado. Execute train() primeiro.")
        
        features = np.array([[irradiancia, temperatura, hora, nebulosidade]])
        prediction = self.model.predict(features)[0]
        return max(0, min(prediction, 1.5))  # Limita entre 0 e capacidade máxima
    
    def predict_day_ahead(self, date=None):
        """Previsão para próximas 24 horas"""
        if date is None:
            date = datetime.now()
        
        predictions = []
        for hora in range(24):
            # Simula condições típicas
            if 6 <= hora <= 18:
                irrad = 800 * np.sin((hora - 6) * np.pi / 12)
                temp = 25 + 5 * np.sin((hora - 6) * np.pi / 12)
                neb = 20
            else:
                irrad = 0
                temp = 22
                neb = 50
            
            power = self.predict(irrad, temp, hora, neb)
            predictions.append({
                'hora': hora,
                'timestamp': date + timedelta(hours=hora),
                'potencia_mw': power,
                'irradiancia': irrad,
                'temperatura': temp
            })
        
        return pd.DataFrame(predictions)


# ========== CLASSE: ANÁLISE DE RISCO ==========
class RiskAnalysis:
    """Análise de riscos e violações na rede"""
    
    def __init__(self, grid_manager):
        self.grid = grid_manager
        self.voltage_limits = {'min': 0.93, 'max': 1.05}
        self.current_limits = {'max': 0.9}  # 90% da capacidade
    
    def check_voltage_violations(self):
        """Verifica violações de tensão"""
        if self.grid.net is None:
            return pd.DataFrame()
        
        buses = self.grid.net.res_bus.copy()
        violations = buses[
            (buses.vm_pu > self.voltage_limits['max']) | 
            (buses.vm_pu < self.voltage_limits['min'])
        ].copy()
        
        if not violations.empty:
            violations['violation_type'] = violations.apply(
                lambda x: 'Sobretensão' if x.vm_pu > self.voltage_limits['max'] else 'Subtensão',
                axis=1
            )
            violations['severity'] = violations.apply(
                lambda x: abs(x.vm_pu - 1.0) * 100,
                axis=1
            )
        
        return violations
    
    def check_line_loading(self):
        """Verifica carregamento das linhas"""
        if self.grid.net is None:
            return pd.DataFrame()
        
        lines = self.grid.net.res_line.copy()
        overloads = lines[lines.loading_percent > self.current_limits['max'] * 100].copy()
        
        if not overloads.empty:
            overloads['severity'] = overloads['loading_percent'] - 90
        
        return overloads
    
    def calculate_risk_score(self):
        """Calcula score de risco global"""
        voltage_viol = self.check_voltage_violations()
        line_overload = self.check_line_loading()
        
        risk_score = 0
        risk_factors = []
        
        if not voltage_viol.empty:
            risk_score += len(voltage_viol) * 30
            risk_factors.append(f"{len(voltage_viol)} violação(ões) de tensão")
        
        if not line_overload.empty:
            risk_score += len(line_overload) * 40
            risk_factors.append(f"{len(line_overload)} linha(s) sobrecarregada(s)")
        
        risk_level = "BAIXO"
        if risk_score > 50:
            risk_level = "MÉDIO"
        if risk_score > 100:
            risk_level = "ALTO"
        
        return {
            'score': min(risk_score, 100),
            'level': risk_level,
            'factors': risk_factors
        }
    
    def generate_report(self):
        """Gera relatório completo de análise"""
        report = {
            'timestamp': datetime.now(),
            'voltage_violations': self.check_voltage_violations(),
            'line_overloads': self.check_line_loading(),
            'risk_assessment': self.calculate_risk_score()
        }
        return report


# ========== CLASSE: VISUALIZAÇÃO ==========
class NetworkVisualizer:
    """Visualização da rede e dados"""
    
    @staticmethod
    def plot_voltage_profile(grid_manager):
        """Plota perfil de tensão"""
        if grid_manager.net is None:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        buses = grid_manager.net.res_bus
        
        ax.plot(buses.index, buses.vm_pu, 'o-', linewidth=2, markersize=8, label='Tensão')
        ax.axhline(y=1.05, color='r', linestyle='--', label='Limite Superior')
        ax.axhline(y=0.93, color='r', linestyle='--', label='Limite Inferior')
        ax.axhline(y=1.0, color='g', linestyle=':', label='Referência')
        
        ax.set_xlabel('Barra', fontsize=12)
        ax.set_ylabel('Tensão (pu)', fontsize=12)
        ax.set_title('Perfil de Tensão da Rede', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig
    
    @staticmethod
    def plot_generation_forecast(forecast_df):
        """Plota previsão de geração"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Gráfico de potência
        ax1.plot(forecast_df.hora, forecast_df.potencia_mw, 'b-', linewidth=2)
        ax1.fill_between(forecast_df.hora, 0, forecast_df.potencia_mw, alpha=0.3)
        ax1.set_ylabel('Potência (MW)', fontsize=11)
        ax1.set_title('Previsão de Geração Solar - 24h', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Gráfico de condições
        ax2.plot(forecast_df.hora, forecast_df.irradiancia, 'orange', label='Irradiância')
        ax2.set_xlabel('Hora do Dia', fontsize=11)
        ax2.set_ylabel('Irradiância (W/m²)', fontsize=11)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(forecast_df.hora, forecast_df.temperatura, 'red', label='Temperatura')
        ax2_twin.set_ylabel('Temperatura (°C)', fontsize=11)
        
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# ========== INTERFACE STREAMLIT ==========
def create_streamlit_app():
    """Cria interface web interativa"""
    
    st.set_page_config(
        page_title="Gestão Inteligente MMGD",
        page_icon="⚡",
        layout="wide"
    )
    
    st.title("⚡ Sistema de Gestão Inteligente de MMGD")
    st.markdown("**Monitoramento e Previsão de Microgeração Distribuída**")
    st.markdown("---")
    
    # Inicialização
    if 'grid_manager' not in st.session_state:
        st.session_state.grid_manager = GridManager()
        st.session_state.grid_manager.create_network()
        st.session_state.forecast = GenerationForecast()
        st.session_state.forecast.train()
        st.session_state.risk_analyzer = RiskAnalysis(st.session_state.grid_manager)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        st.subheader("Condições Atuais")
        irradiancia = st.slider("Irradiância (W/m²)", 0, 1000, 800)
        temperatura = st.slider("Temperatura (°C)", 15, 40, 28)
        hora = st.slider("Hora do Dia", 0, 23, 12)
        nebulosidade = st.slider("Nebulosidade (%)", 0, 100, 20)
        
        st.subheader("Carga Industrial")
        carga_p = st.slider("Potência Ativa (MW)", 0.5, 2.0, 1.2, 0.1)
        carga_q = st.slider("Potência Reativa (MVAr)", 0.2, 1.0, 0.4, 0.1)
        
        if st.button("🔄 Atualizar Simulação", type="primary"):
            st.rerun()
    
    # Previsão de geração
    power_pred = st.session_state.forecast.predict(
        irradiancia, temperatura, hora, nebulosidade
    )
    
    # Atualiza rede
    st.session_state.grid_manager.update_generation(power_pred)
    st.session_state.grid_manager.update_load(0, carga_p, carga_q)
    success, msg = st.session_state.grid_manager.run_power_flow()
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💡 Geração Solar", f"{power_pred:.2f} MW", 
                  delta=f"{(power_pred/1.5)*100:.1f}% capacidade")
    
    with col2:
        st.metric("🏭 Carga Industrial", f"{carga_p:.2f} MW")
    
    with col3:
        if success:
            risk = st.session_state.risk_analyzer.calculate_risk_score()
            st.metric("⚠️ Risco Operacional", risk['level'], 
                      delta=f"Score: {risk['score']}")
        else:
            st.metric("⚠️ Status", "ERRO", delta="Não convergiu")
    
    with col4:
        if success and len(st.session_state.grid_manager.net.res_bus) > 0:
            avg_voltage = st.session_state.grid_manager.net.res_bus.vm_pu.mean()
            st.metric("📊 Tensão Média", f"{avg_voltage:.3f} pu")
    
    # Tabs principais
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Análise de Rede", "🔮 Previsões", "⚠️ Riscos", "📊 Métricas ML"])
    
    with tab1:
        if success:
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("Perfil de Tensão")
                fig = NetworkVisualizer.plot_voltage_profile(st.session_state.grid_manager)
                st.pyplot(fig)
            
            with col_b:
                st.subheader("Resultados das Barras")
                st.dataframe(
                    st.session_state.grid_manager.net.res_bus[['vm_pu', 'va_degree', 'p_mw', 'q_mvar']],
                    use_container_width=True
                )
                
                st.subheader("Carregamento de Linhas")
                st.dataframe(
                    st.session_state.grid_manager.net.res_line[['loading_percent', 'i_ka']],
                    use_container_width=True
                )
        else:
            st.error(f"❌ {msg}")
    
    with tab2:
        st.subheader("Previsão de Geração - Próximas 24h")
        forecast_24h = st.session_state.forecast.predict_day_ahead()
        
        fig = NetworkVisualizer.plot_generation_forecast(forecast_24h)
        st.pyplot(fig)
        
        st.subheader("Dados da Previsão")
        st.dataframe(forecast_24h, use_container_width=True)
    
    with tab3:
        st.subheader("Análise de Riscos")
        
        if success:
            report = st.session_state.risk_analyzer.generate_report()
            risk_data = report['risk_assessment']
            
            # Score visual
            col_x, col_y = st.columns([1, 2])
            with col_x:
                st.metric("Score de Risco", f"{risk_data['score']}/100", 
                          delta=risk_data['level'])
            
            # Violações
            voltage_viol = report['voltage_violations']
            if not voltage_viol.empty:
                st.error("⚠️ Violações de Tensão Detectadas")
                st.dataframe(voltage_viol, use_container_width=True)
            else:
                st.success("✅ Sem violações de tensão")
            
            line_overload = report['line_overloads']
            if not line_overload.empty:
                st.warning("⚠️ Linhas Sobrecarregadas")
                st.dataframe(line_overload, use_container_width=True)
            else:
                st.success("✅ Linhas operando normalmente")
            
            if risk_data['factors']:
                st.info("**Fatores de Risco:**\n" + "\n".join([f"- {f}" for f in risk_data['factors']]))
        else:
            st.error("Não foi possível realizar análise de risco")
    
    with tab4:
        st.subheader("Métricas do Modelo de Machine Learning")
        
        if st.session_state.forecast.is_trained:
            metrics = st.session_state.forecast.metrics
            
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("MAE", f"{metrics['mae']:.4f} MW")
            with col_m2:
                st.metric("RMSE", f"{metrics['rmse']:.4f} MW")
            with col_m3:
                st.metric("R² Score", f"{metrics['r2']:.4f}")
            
            st.subheader("Importância das Features")
            importance_df = st.session_state.forecast.feature_importance
            st.bar_chart(importance_df.set_index('feature'))
            st.dataframe(importance_df, use_container_width=True)
        else:
            st.warning("Modelo não treinado")


# ========== EXECUÇÃO PRINCIPAL ==========
if __name__ == "__main__":
    # Modo Streamlit
    try:
        create_streamlit_app()
    except Exception as e:
        print(f"Erro ao executar interface Streamlit: {e}")
        print("\n" + "="*60)
        print("EXECUTANDO MODO TESTE (CLI)")
        print("="*60 + "\n")
        
        # Modo teste via CLI
        print("1. Criando rede elétrica...")
        grid = GridManager()
        grid.create_network()
        print("   ✓ Rede criada com sucesso\n")
        
        print("2. Treinando modelo de previsão...")
        forecast = GenerationForecast()
        metrics = forecast.train()
        print(f"   ✓ Modelo treinado | R²: {metrics['r2']:.4f} | MAE: {metrics['mae']:.4f} MW\n")
        
        print("3. Fazendo previsão...")
        power = forecast.predict(irradiancia=850, temperatura=30, hora=12, nebulosidade=15)
        print(f"   ✓ Previsão: {power:.2f} MW\n")
        
        print("4. Atualizando rede e executando fluxo de potência...")
        grid.update_generation(power)
        success, msg = grid.run_power_flow()
        print(f"   ✓ {msg}\n")
        
        if success:
            print("5. Análise de riscos...")
            risk = RiskAnalysis(grid)
            risk_report = risk.generate_report()
            print(f"   ✓ Nível de risco: {risk_report['risk_assessment']['level']}\n")
            
            print("6. Resultados das barras:")
            print(grid.net.res_bus[['vm_pu', 'p_mw', 'q_mvar']])
            
            print("\n" + "="*60)
            print("TESTE CONCLUÍDO COM SUCESSO!")
            print("="*60)




