"""BotTrainer — Page 2: Model Trainer"""
import sys, time, pandas as pd, streamlit as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.nlu_engine import train_model, ALGORITHMS

st.set_page_config(page_title="Model Trainer", page_icon="⚡", layout="wide")
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#0E1117;}
[data-testid="stSidebar"]{background:#0A0F1A;border-right:1px solid #1E2A3A;}
[data-testid="metric-container"]{background:#131C2E;border:1px solid #1E2A3A;border-radius:10px;padding:16px!important;}
h1{color:#00C9A7!important;}h2,h3{color:#E8ECF4!important;}
.stProgress>div>div{background:#00C9A7!important;}
</style>""", unsafe_allow_html=True)

if "train_result" not in st.session_state: st.session_state.train_result = None

st.title("⚡ Model Trainer")
st.markdown("Configure hyperparameters and train your NLU intent classifier.")
st.markdown("---")

if "dataset" not in st.session_state or st.session_state.dataset is None:
    st.warning("⚠️ No dataset loaded. Go to **📁 Dataset Manager** first.")
    st.stop()

df: pd.DataFrame = st.session_state.dataset
st.success(f"✅ Dataset ready: **{len(df)}** samples · **{df['intent'].nunique()}** intents")
st.markdown("---")

# ── Config ─────────────────────────────────────────────────────────────────
cc, ci = st.columns([1,1])
with cc:
    st.markdown("#### ⚙️ Configuration")
    algo = st.selectbox("Algorithm", list(ALGORITHMS.keys()))
    test_size = st.slider("Test Split", 0.1, 0.4, 0.2, 0.05,
                          help=f"Train: {int(len(df)*0.8)} | Test: {int(len(df)*0.2)}")
    st.caption(f"Train: **{int(len(df)*(1-test_size))}**  |  Test: **{int(len(df)*test_size)}** samples")
    ngmin, ngmax = st.select_slider("N-gram Range", [1,2,3], value=(1,2))
    max_feat = st.select_slider("Max TF-IDF Features", [500,1000,2000,5000,10000], value=5000)
    rm_sw = st.checkbox("Remove Stopwords", value=False)
    seed  = st.number_input("Random Seed", value=42, step=1)

with ci:
    st.markdown("#### 📘 Algorithm Guide")
    guides = {
        "Logistic Regression": "⭐ **Best all-rounder.** Fast, interpretable, excellent for NLU multi-class. Recommended default.",
        "Linear SVM":          "⚔️ **Great precision.** Finds optimal margin decision boundary. Very fast with calibrated probabilities.",
        "Naive Bayes":         "⚡ **Fastest to train.** Probabilistic baseline. Best when data is limited.",
        "Random Forest":       "🌲 **Robust ensemble.** Handles noisy/imbalanced data well. Slower inference.",
        "Gradient Boosting":   "🚀 **Highest accuracy.** Sequential boosting. Best with clean, well-labelled data.",
    }
    st.info(guides[algo])
    st.markdown("#### TF-IDF Summary")
    st.markdown(f"""
| Setting | Value |
|---|---|
| N-gram Range | `({ngmin}, {ngmax})` |
| Max Features | `{max_feat:,}` |
| Sublinear TF | `True` |
| Stopwords Removed | `{'Yes' if rm_sw else 'No'}` |
""")

st.markdown("---")
train_btn = st.button("⚡ TRAIN MODEL", type="primary", use_container_width=True)

if train_btn:
    st.markdown("---")
    st.markdown("#### 🔄 Training Log")
    bar   = st.progress(0)
    log_box = st.empty()
    lines = []

    def log(msg):
        lines.append(f"`{time.strftime('%H:%M:%S')}` {msg}")
        log_box.markdown("\n\n".join(lines))

    log("🔧 Initializing pipeline...")
    bar.progress(10); time.sleep(0.15)
    log(f"📚 Preparing TF-IDF (ngram=({ngmin},{ngmax}), max_features={max_feat:,})...")
    bar.progress(30); time.sleep(0.2)
    log(f"⚖️  Split: {int(len(df)*(1-test_size))} train / {int(len(df)*test_size)} test...")
    bar.progress(50); time.sleep(0.15)
    log(f"🧠 Training `{algo}`...")
    bar.progress(65)

    try:
        res = train_model(df, algo=algo, test_size=test_size,
                          ngram=(ngmin,ngmax), max_feat=max_feat,
                          rm_sw=rm_sw, seed=int(seed))
        bar.progress(85)
        log("📊 Running 5-fold cross-validation...")
        time.sleep(0.2); bar.progress(100)
        log(f"✅ **Accuracy: {res['accuracy']*100:.2f}%** | CV: {res['cv_acc'].mean()*100:.2f}% ±{res['cv_acc'].std()*100:.2f}% | Time: {res['train_time']*1000:.0f}ms")
        st.session_state.train_result = res
        st.balloons()
    except Exception as e:
        import traceback
        st.error(f"❌ Training failed: {e}")
        st.code(traceback.format_exc())
        st.stop()

# ── Results ────────────────────────────────────────────────────────────────
if st.session_state.train_result:
    res = st.session_state.train_result
    st.markdown("---")
    st.markdown("#### 🎯 Results")
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Accuracy",    f"{res['accuracy']*100:.2f}%")
    m2.metric("CV Mean",     f"{res['cv_acc'].mean()*100:.2f}%")
    m3.metric("CV Std",      f"±{res['cv_acc'].std()*100:.2f}%")
    m4.metric("Train Samples", res["train_n"])
    m5.metric("Train Time",  f"{res['train_time']*1000:.0f}ms")

    rpt = res["report"]
    rows = [{"Intent": c, "Precision": f"{rpt[c]['precision']:.3f}",
             "Recall": f"{rpt[c]['recall']:.3f}", "F1": f"{rpt[c]['f1-score']:.3f}",
             "Support": int(rpt[c]["support"])}
            for c in res["classes"] if c in rpt]
    st.markdown("**Per-Intent Report**")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    import joblib, io
    buf = io.BytesIO()
    joblib.dump({"pipeline": res["pipeline"], "label_encoder": res["label_encoder"],
                 "classes": res["classes"], "algo": res["algo"]}, buf)
    buf.seek(0)
    st.download_button("⬇️ Download Trained Model (.joblib)", data=buf,
                       file_name="nlu_model.joblib", use_container_width=True)
