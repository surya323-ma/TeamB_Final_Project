"""BotTrainer NLU — Home"""
import streamlit as st, json, pandas as pd
from pathlib import Path

st.set_page_config(page_title="BotTrainer NLU", page_icon="🤖", layout="wide",initial_sidebar_state="expanded")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#0E1117; }
[data-testid="stSidebar"] { background:#0A0F1A; border-right:1px solid #1E2A3A; }
[data-testid="metric-container"] {
    background:#131C2E; border:1px solid #1E2A3A;
    border-radius:10px; padding:16px !important; }
h1{color:#00C9A7!important;letter-spacing:2px;}
h2,h3{color:#E8ECF4!important;}
.stButton>button{background:#131C2E;border:1px solid #1E2A3A;color:#E8ECF4;border-radius:8px;}
.stButton>button:hover{border-color:#00C9A7;color:#00C9A7;}
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🤖 BotTrainer NLU")
    st.markdown("---")
    if "train_result" in st.session_state and st.session_state.train_result:
        r = st.session_state.train_result
        st.success(f"Model ACTIVE\n\n**{r['algo']}**\nAccuracy: {r['accuracy']*100:.1f}%")
    else:
        st.warning("No model trained yet.")
    st.markdown("---")
    st.caption("BotTrainer v2.0  ·  Streamlit + scikit-learn")

st.title("🤖 BOTTRAINER  NLU")
st.markdown("### Natural Language Understanding — Model Trainer & Evaluator")
st.markdown("🤖🤖")

c1,c2,c3,c4,c5 = st.columns(5)
c1.info("**📁 Dataset**\nLoad & manage training data")
c2.info("**⚡ Trainer**\nTrain intent classifier")
c3.info("**📊 Evaluator**\nMetrics & confusion matrix")
c4.info("**🔮 Live Test**\nReal-time predictions")
c5.info("**📈 Compare**\nBenchmark algorithms")

st.markdown("---")
col_a, col_b = st.columns([2,1])
with col_a:
    st.markdown("""
#### Features
- **5 ML algorithms**: Logistic Regression · Linear SVM · Naive Bayes · Random Forest · Gradient Boosting  
- **TF-IDF** with configurable n-gram range & vocabulary size  
- **Rule-based entity extraction**: DATE · TIME · LOCATION · PHONE · CURRENCY · EMAIL  
- **Evaluation**: Accuracy · Precision · Recall · F1 · Confusion Matrix · 5-Fold Cross-Validation  
- **Model export**: Download as `.joblib` for production use  
""")
with col_b:
    st.markdown("#### Quick Start")
    st.markdown("1. **📁 Dataset Manager** — load data\n2. **⚡ Trainer** — train model\n3. **📊 Evaluator** — check metrics\n4. **🔮 Live Test** — test messages")
    st.markdown("---")
    if st.button("🚀 Load Sample Dataset & Go", use_container_width=True):
        p = Path(__file__).parent / "data" / "sample_dataset.json"
        data = json.loads(p.read_text())
        st.session_state.dataset = pd.DataFrame(data)
        st.success(f"✅ Loaded {len(data)} samples, {pd.DataFrame(data)['intent'].nunique()} intents")
