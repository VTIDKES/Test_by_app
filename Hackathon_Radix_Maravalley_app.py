import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Hackathon Energia • VTS", layout="wide")

# =====================================
#           MENU
# =====================================
st.sidebar.title("⚡ Hackathon — Soluções para Energia")
menu = st.sidebar.selectbox(
    "Selecione uma solução:",
    [
        "📉 Monitoramento de Perdas e Fraudes",
        "🔆 Previsão de Geração Solar",
        "🏭 Digital Twin de Subestação / Solar"
    ]
)

# =====================================
#   1) MONITORAMENTO DE FRAUDES
# =====================================
if menu == "📉 Monitoramento de Perdas e Fraudes":
    st.title("📉 Monitoramento Inteligente de Perdas e Fraudes")
    st.write("Detecção automática de anomalias via Isolation Forest (sklearn).")

    file = st.file_uploader("Upload CSV com consumo", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.write("Pré-visualização:")
        st.dataframe(df.head())

        if "consumption" not in df.columns:
            st.error("Arquivo precisa conter a coluna 'consumption'")
        else:
            cont = st.slider("Nível de sensibilidade do detector", 0.001, 0.2, 0.05)

            if st.button("Detectar Fraudes"):
                model = IsolationForest(contamination=cont, random_state=42)
                df["anomaly"] = model.fit_predict(df[["consumption"]])
                df["anomaly"] = df["anomaly"].apply(lambda x: 1 if x == -1 else 0)

                st.metric("Total de anomalias detectadas", int(df["anomaly"].sum()))

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=df["consumption"],
                    mode="lines",
                    name="Consumo"
                ))
                fig.add_trace(go.Scatter(
                    y=df.loc[df["anomaly"] == 1, "consumption"],
                    mode="markers",
                    name="Anomalias"
                ))
                st.plotly_chart(fig, use_container_width=True)

                st.download_button(
                    "Baixar resultados",
                    df.to_csv(index=False).encode("utf-8"),
                    "resultados_fraudes.csv"
                )


# =====================================
#   2) PREVISÃO DE GERAÇÃO SOLAR
# =====================================
elif menu == "🔆 Previsão de Geração Solar":
    st.title("🔆 Previsão Inteligente de Geração Solar")
    st.write("Modelo simples baseado em regressão linear.")

    file = st.file_uploader("Upload CSV com irradiância/temperatura", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.write(df.head())

        if {"irradiance", "temperature", "power"}.issubset(df.columns):
            if st.button("Treinar Modelo"):
                X = df[["irradiance", "temperature"]]
                y = df["power"]

                model = LinearRegression()
                model.fit(X, y)

                st.success("Modelo treinado!")

                irr = st.slider("Irradiância (W/m²)", 0, 1200, 800)
                temp = st.slider("Temperatura (°C)", -10, 80, 35)

                pred = model.predict([[irr, temp]])[0]

                st.metric("Geração Estimada (kW)", f"{pred:.2f}")

                fig = go.Figure(go.Indicator(
                    mode="number+gauge",
                    value=pred,
                    title={"text": "Potência"},
                    gauge={"axis": {"range": [0, max(1000, pred*1.5)]}}
                ))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("O CSV deve conter: irradiance, temperature, power")


# =====================================
#         3) DIGITAL TWIN
# =====================================
elif menu == "🏭 Digital Twin de Subestação / Solar":
    st.title("🏭 Digital Twin da Subestação / Usina Solar")

    tensao = st.slider("Tensão (kV)", 10, 500, 69)
    corrente = st.slider("Corrente (A)", 0, 3000, 450)
    temperatura = st.slider("Temperatura (°C)", -10, 120, 45)

    potencia = (tensao * 1000 * corrente) / (np.sqrt(3) * 1000)

    st.metric("Potência Aparente (MVA)", f"{potencia/1e6:.3f}")

    fig = go.Figure(go.Indicator(
        mode="number+gauge",
        value=temperatura,
        title={"text": "Temperatura"},
        gauge={"axis": {"range": [-10, 120]}}
    ))
    st.plotly_chart(fig, use_container_width=True)


