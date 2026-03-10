"""BotTrainer — Page 5: Compare Algorithms"""
import sys, pandas as pd, streamlit as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.nlu_engine import compare_all
from utils.charts import algo_compare

st.set_page_config(page_title="Compare Algorithms", page_icon="📈", layout="wide")
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#0E1117;}
[data-testid="stSidebar"]{background:#0A0F1A;border-right:1px solid #1E2A3A;}
[data-testid="metric-container"]{background:#131C2E;border:1px solid #1E2A3A;border-radius:10px;padding:16px!important;}
h1{color:#00C9A7!important;}h2,h3{color:#E8ECF4!important;}
</style>""", unsafe_allow_html=True)

st.title("📈 Compare Algorithms")
st.markdown("Benchmark all 5 ML algorithms on the same dataset and find the best performer.")
st.markdown("---")

if "dataset" not in st.session_state or st.session_state.dataset is None:
    st.warning("⚠️ No dataset loaded. Go to **📁 Dataset Manager** first.")
    st.stop()

df = st.session_state.dataset
st.info(f"Dataset: **{len(df)}** samples · **{df['intent'].nunique()}** intents")

cc, cr = st.columns([2,1])
with cc:
    test_size = st.slider("Test Split", 0.1, 0.4, 0.2, 0.05)
with cr:
    st.markdown("")
    st.markdown("")
    run = st.button("🚀 Run Benchmark", type="primary", use_container_width=True)

if run:
    with st.spinner("⏳ Training & evaluating all 5 algorithms..."):
        cdf = compare_all(df, test_size=test_size)
    st.session_state.compare_df = cdf

if "compare_df" in st.session_state:
    cdf = st.session_state.compare_df
    st.markdown("---")

    if "Accuracy" in cdf.columns:
        best = cdf.loc[cdf["Accuracy"].idxmax()]
        st.success(f"🏆 Best: **{best['Algorithm']}** — Accuracy `{best['Accuracy']*100:.2f}%`")

    st.markdown("#### 📊 Performance Chart")
    st.plotly_chart(algo_compare(cdf), use_container_width=True)

    st.markdown("#### 📋 Full Results")
    disp = cdf.copy()
    for col in ["Accuracy","Precision","Recall","F1","CV Mean","CV Std"]:
        if col in disp.columns:
            disp[col] = disp[col].apply(lambda x: f"{x*100:.2f}%")
    st.dataframe(disp, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 🥇 Rankings")
    raw = st.session_state.compare_df
    rc1,rc2,rc3 = st.columns(3)
    for col_w, metric in zip([rc1,rc2,rc3], ["Accuracy","F1","CV Mean"]):
        if metric in raw.columns:
            ranked = raw[["Algorithm",metric]].sort_values(metric,ascending=False).reset_index(drop=True)
            ranked.index += 1
            ranked[metric] = ranked[metric].apply(lambda x: f"{x*100:.2f}%")
            col_w.markdown(f"**{metric}**")
            col_w.dataframe(ranked, use_container_width=True)

    st.markdown("---")
    best_algo = raw.loc[raw["Accuracy"].idxmax(),"Algorithm"]
    recs = {
        "Logistic Regression": "Great all-around. Fast, interpretable, recommended default for NLU.",
        "Linear SVM":          "Best for high-precision intent separation on medium datasets.",
        "Naive Bayes":         "Fastest training. Use as a quick baseline or when data is very limited.",
        "Random Forest":       "Best when data is noisy or intents are imbalanced.",
        "Gradient Boosting":   "Highest raw accuracy potential. Needs clean, sufficient data.",
    }
    st.info(f"💡 **Recommendation: {best_algo}**\n\n{recs.get(best_algo,'')}")
    st.download_button("⬇️ Download Report", raw.to_csv(index=False),
                       "algo_comparison.csv","text/csv")
