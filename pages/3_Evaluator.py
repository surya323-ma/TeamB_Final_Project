"""BotTrainer — Page 3: Evaluator"""
import sys, pandas as pd, streamlit as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.charts import confusion_heatmap, per_intent_bars, radar, cv_bars, conf_dist

st.set_page_config(page_title="Evaluator", page_icon="📊", layout="wide")
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#0E1117;}
[data-testid="stSidebar"]{background:#0A0F1A;border-right:1px solid #1E2A3A;}
[data-testid="metric-container"]{background:#131C2E;border:1px solid #1E2A3A;border-radius:10px;padding:16px!important;}
h1{color:#00C9A7!important;}h2,h3{color:#E8ECF4!important;}
</style>""", unsafe_allow_html=True)

st.title("📊 Model Evaluator")
st.markdown("Deep-dive metrics: accuracy, confusion matrix, F1, cross-validation, confidence analysis.")
st.markdown("---")

if "train_result" not in st.session_state or not st.session_state.train_result:
    st.warning("⚠️ No trained model. Go to **⚡ Trainer** first.")
    st.stop()

res = st.session_state.train_result
rpt     = res["report"]
classes = res["classes"]
macro   = rpt.get("macro avg", {})
weighted = rpt.get("weighted avg", {})

# ── KPIs ──────────────────────────────────────────────────────────────────
st.markdown("#### 🎯 Overall Metrics")
k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("Accuracy",          f"{res['accuracy']*100:.2f}%")
k2.metric("Macro Precision",   f"{macro.get('precision',0)*100:.2f}%")
k3.metric("Macro Recall",      f"{macro.get('recall',0)*100:.2f}%")
k4.metric("Macro F1",          f"{macro.get('f1-score',0)*100:.2f}%")
k5.metric("CV Accuracy (mean)",f"{res['cv_acc'].mean()*100:.2f}%")
k6.metric("CV Std",            f"±{res['cv_acc'].std()*100:.2f}%")

st.markdown("---")
# ── Confusion Matrix + Radar ──────────────────────────────────────────────
col1, col2 = st.columns([3,2])
with col1:
    st.markdown("#### 🔲 Confusion Matrix")
    st.plotly_chart(confusion_heatmap(res["cm"], classes), use_container_width=True)
with col2:
    st.markdown("#### 🕸️ F1 Radar")
    st.plotly_chart(radar(rpt, classes), use_container_width=True)

st.markdown("---")
# ── Per-intent bar ─────────────────────────────────────────────────────────
st.markdown("#### 📐 Per-Intent Metrics")
st.plotly_chart(per_intent_bars(rpt, classes), use_container_width=True)

rows = []
for c in classes:
    if c not in rpt: continue
    p,r,f1 = rpt[c]["precision"], rpt[c]["recall"], rpt[c]["f1-score"]
    status = "✅ Good" if f1>=0.8 else "⚠️ Fair" if f1>=0.5 else "❌ Poor"
    rows.append({"Intent":c,"Precision":round(p,4),"Recall":round(r,4),
                 "F1":round(f1,4),"Support":int(rpt[c]["support"]),"Status":status})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("---")
col_cv, col_conf = st.columns(2)
with col_cv:
    st.markdown("#### 🔁 Cross-Validation (5-fold)")
    st.plotly_chart(cv_bars(res["cv_acc"]), use_container_width=True)
    cv_df = pd.DataFrame({"Fold":[f"Fold {i+1}" for i in range(len(res["cv_acc"]))],
                           "Accuracy":[f"{s:.4f}" for s in res["cv_acc"]],
                           "F1 Macro":[f"{s:.4f}" for s in res["cv_f1"]]})
    st.dataframe(cv_df, use_container_width=True, hide_index=True)

with col_conf:
    st.markdown("#### 📈 Confidence Distribution")
    pred_df = res["pred_df"]
    if pred_df["confidence"].notna().any():
        st.plotly_chart(conf_dist(pred_df), use_container_width=True)
        c_ok  = pred_df[pred_df.correct]["confidence"].mean()
        c_bad = pred_df[~pred_df.correct]["confidence"].mean()
        st.metric("Avg confidence (correct)",   f"{c_ok*100:.1f}%"  if pd.notna(c_ok)  else "N/A")
        st.metric("Avg confidence (incorrect)", f"{c_bad*100:.1f}%" if pd.notna(c_bad) else "N/A")
    else:
        st.info("No probability scores for this algorithm.")

st.markdown("---")
# ── Predictions table ──────────────────────────────────────────────────────
st.markdown("#### 🔍 Test Set Predictions")
pf = res["pred_df"].copy()
pf["✓"] = pf["correct"].map({True:"✅",False:"❌"})
if pf["confidence"].notna().any():
    pf["Confidence"] = pf["confidence"].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")

fc1,fc2 = st.columns(2)
with fc1: show = st.selectbox("Show",["All","Correct Only","Incorrect Only"])
with fc2: int_f = st.selectbox("Filter Intent",["All"]+sorted(classes))

disp = pf.copy()
if show == "Correct Only":   disp = disp[disp["correct"]]
if show == "Incorrect Only": disp = disp[~disp["correct"]]
if int_f != "All": disp = disp[disp["actual"]==int_f]
cols = ["✓","text","actual","predicted"]+["Confidence"]*(pf["confidence"].notna().any())
st.dataframe(disp[cols].reset_index(drop=True), use_container_width=True, height=300, hide_index=True)
st.caption(f"{len(disp)} of {len(pf)} predictions  |  {pf['correct'].sum()}/{len(pf)} correct")

st.markdown("---")
rep_df = pd.DataFrame(rows)
st.download_button("⬇️ Download Metrics Report (CSV)", rep_df.to_csv(index=False),
                   "evaluation_report.csv","text/csv")
