# ========================================
# IBOVESPA PREDICTION DASHBOARD
# TECH CHALLENGE - FASE 4
# VERSÃO CONSERVADORA + COMPATÍVEL COM O MODELO
# ========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

# ========================================
# CONFIG
# ========================================

st.set_page_config(page_title="IBOVESPA Dashboard", layout="wide")
st.title("📊 IBOVESPA Prediction Dashboard")

# ========================================
# LOAD DATA
# ========================================

@st.cache_data(ttl=3600)
def load_csv():
    return pd.read_csv(
        "Unified_Data.csv",
        parse_dates=["date"]
    )

def create_features(df):
    df = df.copy()

    # Retorno
    df["returns"] = df["close"].pct_change()

    # Lags de retorno
    for i in range(1, 11):
        df[f"returns_lag{i}"] = df["returns"].shift(i)

    # Sinais de retorno
    for i in [1, 2, 3, 5, 10]:
        df[f"sinal_t{i}"] = (df["returns"].rolling(i).mean() > 0).astype(int)

    # Médias móveis
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()

    df["sinal_ma5_ma20"] = (df["ma5"] > df["ma20"]).astype(int)
    df["close_acima_ma5"] = (df["close"] > df["ma5"]).astype(int)
    df["close_acima_ma20"] = (df["close"] > df["ma20"]).astype(int)

    # USD
    if "usd_close" in df.columns:
        df["usd_change"] = df["usd_close"].pct_change()
        df["sinal_usd_up"] = (df["usd_change"] > 0).astype(int)
    else:
        df["usd_change"] = 0
        df["sinal_usd_up"] = 0

    # Selic
    if "selic" in df.columns:
        df["selic_change"] = df["selic"].diff()
        df["selic_subindo"] = (df["selic_change"] > 0).astype(int)
    else:
        df["selic_change"] = 0
        df["selic_subindo"] = 0

    return df.dropna()

@st.cache_data(ttl=3600)
def load_features():
    df = load_csv()
    df_feat = create_features(df)
    return df, df_feat

# ========================================
# LOAD MODEL
# ========================================

@st.cache_data(ttl=3600)
def load_model():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("feature_columns.json") as f:
        feature_columns = json.load(f)["feature_columns"]

    with open("model_info.json") as f:
        model_info = json.load(f)

    return model, feature_columns, model_info

def predict(df_feat, model, feature_columns):
    X = df_feat[feature_columns].iloc[-1:]
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0].max() * 100
    return ("ALTA" if pred == 1 else "BAIXA"), proba

# ========================================
# MAIN
# ========================================

df, df_feat = load_features()
model, feature_columns, model_info = load_model()

# ========================================
# SIDEBAR (COMO NO ORIGINAL)
# ========================================

st.sidebar.header("⚙️ Período de Análise")
periodo = st.sidebar.selectbox(
    "Selecione o período:",
    ["30 dias", "60 dias", "90 dias", "180 dias", "1 ano", "Todo histórico"]
)

mapa = {
    "30 dias": 30,
    "60 dias": 60,
    "90 dias": 90,
    "180 dias": 180,
    "1 ano": 252,
    "Todo histórico": len(df)
}

df_plot = df.tail(mapa[periodo])

# ========================================
# RESUMO EXECUTIVO (RESTAURADO)
# ========================================

st.subheader("📌 Resumo Executivo")

pred, conf = predict(df_feat, model, feature_columns)

col1, col2, col3 = st.columns(3)

col1.metric("Preço Atual", f"R$ {df_plot['close'].iloc[-1]:,.2f}")
col2.metric("Tendência Prevista", pred)
col3.metric("Confiança do Modelo", f"{conf:.1f}%")

st.markdown("""
**Interpretação:**  
O modelo indica a **direção esperada do IBOVESPA no próximo pregão** com base em
indicadores técnicos, comportamento recente do índice, dólar e taxa Selic.
""")

# ========================================
# TABS
# ========================================

tab1, tab2, tab3 = st.tabs([
    "📈 Série Histórica",
    "📊 Indicadores",
    "🎯 Performance do Modelo"
])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_plot["date"],
        y=df_plot["close"],
        mode="lines",
        name="IBOVESPA"
    ))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.write("Indicadores utilizados no modelo:")
    st.code(feature_columns)

with tab3:
    c1, c2, c3 = st.columns(3)
    c1.metric("Acurácia", f"{model_info['accuracy']:.2%}")
    c2.metric("Precision", f"{model_info['precision']:.2%}")
    c3.metric("Recall", f"{model_info['recall']:.2%}")

# ========================================
# FOOTER
# ========================================

st.caption(
    "⚠️ Previsão direcional para fins educacionais. "
    "Não constitui recomendação de investimento."
)
