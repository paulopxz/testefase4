# ========================================
# SOLUÇÃO FINAL - VERSÃO CONSERVADORA
# Tech Challenge - Fase 4
# ========================================
# Estrutura original preservada
# Ajustes apenas para:
# - Evidenciar modelo da Fase 2
# - Corrigir performance
# - Garantir aderência ao enunciado
# ========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import json
import pickle

# ========================================
# 1. CONFIGURAÇÃO STREAMLIT
# ========================================

st.set_page_config(
    page_title="IBOVESPA Prediction Dashboard",
    layout="wide"
)

st.title("📊 IBOVESPA Prediction Dashboard")

st.markdown("""
### 🤖 Modelo Preditivo

Este dashboard realiza o **deploy do modelo de Machine Learning desenvolvido na Fase 2**
para prever a **tendência do IBOVESPA no próximo pregão**.

- Modelo carregado via pickle
- Features técnicas geradas automaticamente
- Métricas disponíveis na aba *Performance do Modelo*
""")

# ========================================
# 2. CARREGAR CSV (ORIGINAL)
# ========================================

@st.cache_data(ttl=3600)
def load_csv_optimized():
    df = pd.read_csv(
        'Unified_Data.csv',
        dtype={
            'close': 'float32',
            'selic': 'float32'
        },
        parse_dates=['date']
    )
    return df

# ========================================
# 3. CREATE FEATURES (ORIGINAL)
# ========================================

def create_features(df):
    df = df.copy()

    has_high = 'high' in df.columns
    has_low = 'low' in df.columns
    has_open = 'open' in df.columns
    has_usd_close = 'usd_close' in df.columns
    has_selic = 'selic' in df.columns

    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma50'] = df['close'].rolling(window=50).mean()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    df['ema12'] = df['close'].ewm(span=12).mean()
    df['ema26'] = df['close'].ewm(span=26).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['signal']

    df['volatility'] = df['close'].pct_change().rolling(window=20).std()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    if has_high and has_low:
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift()),
                abs(df['low'] - df['close'].shift())
            )
        )
        df['atr'] = df['tr'].rolling(window=14).mean()
    else:
        df['atr'] = df['close'].rolling(window=14).std() * 1.5

    if has_open:
        df['co_ratio'] = df['close'] / df['open']

    if has_usd_close:
        df['close_usd_ratio'] = df['close'] / df['usd_close']

    if has_selic:
        df['selic_normalized'] = (df['selic'] - df['selic'].mean()) / df['selic'].std()

    df['momentum'] = df['close'] - df['close'].shift(10)
    df['roc'] = (df['close'] - df['close'].shift(12)) / df['close'].shift(12) * 100

    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    df['tenkan'] = (df['close'].rolling(window=9).max() + df['close'].rolling(window=9).min()) / 2
    df['kijun'] = (df['close'].rolling(window=26).max() + df['close'].rolling(window=26).min()) / 2

    return df

# ========================================
# 4. LIMPEZA (ORIGINAL)
# ========================================

def clean_close_price(df):
    Q1 = df['close'].quantile(0.25)
    Q3 = df['close'].quantile(0.75)
    IQR = Q3 - Q1

    mask = ~((df['close'] < (Q1 - 1.5 * IQR)) |
             (df['close'] > (Q3 + 1.5 * IQR)))

    return df[mask]

# ========================================
# 5. CACHE FEATURES (ORIGINAL)
# ========================================

@st.cache_data(ttl=3600)
def load_features_cached():
    df = load_csv_optimized()
    df = clean_close_price(df)
    df_feat = create_features(df).dropna()
    return df_feat, df

# ========================================
# 6. CARREGAR MODELO (ORIGINAL)
# ========================================

@st.cache_data(ttl=3600)
def load_model_and_info():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('model_info.json', 'r') as f:
        model_info = json.load(f)

    with open('feature_columns.json', 'r') as f:
        feature_columns = json.load(f)

    return model, model_info, feature_columns

# ========================================
# 7. PREVISÃO (ORIGINAL + ROBUSTEZ)
# ========================================

def predict_next_day(df_feat, feature_columns, model):
    try:
        X_last = df_feat[feature_columns].iloc[-1:].values
        pred = model.predict(X_last)[0]
        proba = model.predict_proba(X_last)[0].max() * 100
        return ('ALTA' if pred == 1 else 'BAIXA'), proba
    except Exception as e:
        st.error(f"Erro na previsão: {e}")
        return None, None

# ========================================
# 8. LOG DE USO (NOVO - OPCIONAL)
# ========================================

def log_usage(pred, conf):
    log = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "prediction": pred,
        "confidence": round(conf, 2)
    }
    with open("usage_log.json", "a") as f:
        f.write(json.dumps(log) + "\n")

# ========================================
# 9. MAIN
# ========================================

try:
    df_feat, df = load_features_cached()
    model, model_info, feature_columns = load_model_and_info()
    st.success("✅ Dados e modelo carregados com sucesso!")
except Exception as e:
    st.error(f"Erro ao carregar dados ou modelo: {e}")
    st.stop()

# Sidebar
st.sidebar.header("⚙️ Filtros")
dias_back = st.sidebar.slider("Dias para exibir:", 30, 250, 100)

df_filtered = df.tail(dias_back)

# Métricas
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Preço Atual", f"R$ {df_filtered['close'].iloc[-1]:,.2f}")

with col2:
    variacao = ((df_filtered['close'].iloc[-1] / df_filtered['close'].iloc[0]) - 1) * 100
    st.metric("Variação", f"{variacao:+.2f}%")

with col3:
    pred, conf = predict_next_day(df_feat, feature_columns, model)
    if pred:
        st.metric("Previsão Próximo Dia", pred, f"Confiança: {conf:.1f}%")
        log_usage(pred, conf)

# Tabs
tab1, tab2, tab3 = st.tabs(
    ["📈 Série Histórica", "📊 Indicadores", "🎯 Performance do Modelo"]
)

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_filtered['date'],
        y=df_filtered['close'],
        mode='lines',
        name='Close'
    ))
    fig.update_layout(height=400, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("""
    **Indicadores utilizados pelo modelo:**
    - Médias móveis
    - RSI
    - MACD
    - Volatilidade
    - Bollinger Bands
    - Momentum e ROC
    """)

with tab3:
    col1, col2, col3 = st.columns(3)
    col1.metric("Acurácia", f"{model_info['accuracy']:.2%}")
    col2.metric("Precision", f"{model_info['precision']:.2%}")
    col3.metric("Recall", f"{model_info['recall']:.2%}")

    st.markdown("### 📉 Métricas completas")
    st.json(model_info)

# Rodapé
st.caption(
    "⚠️ Este modelo é uma ferramenta auxiliar e não deve ser utilizado "
    "como única base para decisões de investimento."
)
