# ========================================
# IBOVESPA PREDICTION DASHBOARD
# Tech Challenge - Fase 4
# VERSÃO FINAL CONSERVADORA
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
# CONFIGURAÇÃO STREAMLIT
# ========================================

st.set_page_config(
    page_title="IBOVESPA Prediction Dashboard",
    layout="wide"
)

st.title("📊 IBOVESPA Prediction Dashboard")

st.markdown("""
### 🤖 Modelo Preditivo (Fase 2)

Este aplicativo realiza o **deploy do modelo de Machine Learning desenvolvido na Fase 2**
para prever a **tendência do IBOVESPA no próximo pregão**.

- Modelo carregado via *pickle*
- Features técnicas automáticas
- Monitoramento das métricas do modelo
""")

# ========================================
# CARGA DE DADOS
# ========================================

@st.cache_data(ttl=3600)
def load_csv_optimized():
    return pd.read_csv(
        "Unified_Data.csv",
        parse_dates=["date"],
        dtype={"close": "float32", "selic": "float32"}
    )

def clean_close_price(df):
    Q1, Q3 = df["close"].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    mask = ~(
        (df["close"] < (Q1 - 1.5 * IQR)) |
        (df["close"] > (Q3 + 1.5 * IQR))
    )
    return df[mask]

def create_features(df):
    df = df.copy()

    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma50"] = df["close"].rolling(50).mean()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    df["ema12"] = df["close"].ewm(span=12).mean()
    df["ema26"] = df["close"].ewm(span=26).mean()
    df["macd"] = df["ema12"] - df["ema26"]
    df["signal"] = df["macd"].ewm(span=9).mean()

    df["volatility"] = df["close"].pct_change().rolling(20).std()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    return df.dropna()

@st.cache_data(ttl=3600)
def load_features_cached():
    df = load_csv_optimized()
    df = clean_close_price(df)
    df_feat = create_features(df)
    return df_feat, df

# ========================================
# MODELO
# ========================================

@st.cache_data(ttl=3600)
def load_model_and_info():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("model_info.json", "r") as f:
        model_info = json.load(f)

    with open("feature_columns.json", "r") as f:
        feature_columns = json.load(f)

    return model, model_info, feature_columns

def extract_feature_columns(feature_columns):
    """
    Trata TODOS os formatos possíveis de feature_columns.json
    """
    if isinstance(feature_columns, dict):
        if "feature_columns" in feature_columns:
            return feature_columns["feature_columns"]
        else:
            return list(feature_columns.keys())
    elif isinstance(feature_columns, list):
        return feature_columns
    else:
        raise ValueError("Formato inválido de feature_columns")

def predict_next_day(df_feat, feature_columns, model):
    try:
        cols = extract_feature_columns(feature_columns)
        X_last = df_feat[cols].iloc[-1:].values

        pred = model.predict(X_last)[0]
        proba = model.predict_proba(X_last)[0].max() * 100

        direction = "ALTA" if pred == 1 else "BAIXA"
        return direction, proba

    except Exception as e:
        st.error(f"Erro na previsão: {e}")
        return None, None

def estimate_next_close(df_feat, last_close):
    """
    Estimativa simples do preço do próximo dia útil
    (não é saída direta do modelo)
    """
    mean_return = df_feat["log_return"].tail(20).mean()
    return last_close * np.exp(mean_return)

# ========================================
# EXECUÇÃO PRINCIPAL
# ========================================

try:
    df_feat, df = load_features_cached()
    model, model_info, feature_columns = load_model_and_info()
    st.success("✅ Dados e modelo carregados com sucesso!")
except Exception as e:
    st.error(f"Erro ao carregar dados ou modelo: {e}")
    st.stop()

# ========================================
# SIDEBAR
# ========================================

st.sidebar.header("⚙️ Filtros")
dias_back = st.sidebar.slider("Período de análise (dias)", 30, 250, 100)
df_filtered = df.tail(dias_back)

# ========================================
# MÉTRICAS PRINCIPAIS
# ========================================

last_close = df_filtered["close"].iloc[-1]
pred, conf = predict_next_day(df_feat, feature_columns, model)
estimated_price = estimate_next_close(df_feat, last_close)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Preço Atual", f"R$ {last_close:,.2f}")

variacao = ((last_close / df_filtered["close"].iloc[0]) - 1) * 100
col2.metric("Variação no Período", f"{variacao:+.2f}%")

if pred:
    col3.metric("Previsão (Modelo)", pred, f"Confiança: {conf:.1f}%")

col4.metric(
    "Preço Estimado (Próx. Dia)",
    f"R$ {estimated_price:,.2f}",
    help="Estimativa estatística baseada no retorno médio recente"
)

# ========================================
# TABS
# ========================================

tab1, tab2, tab3 = st.tabs(
    ["📈 Série Histórica", "📊 Indicadores", "🎯 Performance do Modelo"]
)

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_filtered["date"],
        y=df_filtered["close"],
        mode="lines",
        name="Close"
    ))
    fig.update_layout(height=400, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("""
    **Indicadores utilizados pelo modelo:**

    - Médias móveis
    - RSI
    - MACD
    - Volatilidade
    - Retorno logarítmico
    """)

with tab3:
    st.subheader("📊 Métricas de Validação do Modelo")

    c1, c2, c3 = st.columns(3)

    c1.metric("Acurácia", f"{model_info['accuracy']:.2%}")
    c2.metric("Precision", f"{model_info['precision']:.2%}")
    c3.metric("Recall", f"{model_info['recall']:.2%}")

    with st.expander("🔍 Ver métricas completas"):
        st.json(model_info)

# ========================================
# RODAPÉ
# ========================================

st.caption(
    "⚠️ O modelo fornece uma **previsão direcional**. "
    "O valor estimado é apenas uma aproximação estatística auxiliar."
)
