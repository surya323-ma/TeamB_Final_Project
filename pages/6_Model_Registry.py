"""BotTrainer — Page 6: Model Registry"""
import sys, io, joblib, pandas as pd, streamlit as st
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.nlu_engine import predict

st.set_page_config(page_title="Model Registry", page_icon="🗂️", layout="wide")
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#0E1117;}
[data-testid="stSidebar"]{background:#0A0F1A;border-right:1px solid #1E2A3A;}
[data-testid="metric-container"]{background:#131C2E;border:1px solid #1E2A3A;border-radius:10px;padding:16px!important;}
h1{color:#00C9A7!important;}h2,h3{color:#E8ECF4!important;}
</style>""", unsafe_allow_html=True)

if "registry" not in st.session_state: st.session_state.registry = []

st.title("🗂️ Model Registry")
st.markdown("Save, compare, load and manage all your trained models in one place.")
st.markdown("---")

# ── Save current model ────────────────────────────────────────────────────
if "train_result" in st.session_state and st.session_state.train_result:
    res = st.session_state.train_result
    sc1,sc2,sc3 = st.columns([3,2,1])
    with sc1:
        tag = st.text_input("Model tag / notes", placeholder="e.g. baseline, ngram(1,2), no stopwords")
    with sc3:
        st.markdown(""); st.markdown("")
        if st.button("💾 Save to Registry", type="primary", use_container_width=True):
            entry = {
                "id":       f"model_{len(st.session_state.registry)+1:03d}",
                "algo":     res["algo"],
                "accuracy": res["accuracy"],
                "cv_mean":  float(res["cv_acc"].mean()),
                "f1_macro": float(res["report"].get("macro avg",{}).get("f1-score",0)),
                "intents":  len(res["classes"]),
                "train_n":  res["train_n"],
                "tag":      tag or "—",
                "saved_at": datetime.now().strftime("%H:%M:%S"),
                "pipeline": res["pipeline"],
                "le":       res["label_encoder"],
                "classes":  res["classes"],
            }
            st.session_state.registry.append(entry)
            st.success(f"✅ Saved as `{entry['id']}`")
else:
    st.info("Train a model on the **⚡ Trainer** page to save it here.")

# ── Registry table ─────────────────────────────────────────────────────────
reg = st.session_state.registry
if not reg:
    st.warning("No models saved yet.")
    st.stop()

st.markdown("---")
st.markdown("#### All Saved Models")
rows = [{"ID":e["id"],"Algorithm":e["algo"],
         "Accuracy":f"{e['accuracy']*100:.2f}%","CV Mean":f"{e['cv_mean']*100:.2f}%",
         "Macro F1":f"{e['f1_macro']:.3f}","Intents":e["intents"],
         "Train Samples":e["train_n"],"Tag":e["tag"],"Saved":e["saved_at"]}
        for e in sorted(reg, key=lambda x: -x["accuracy"])]
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("---")
# ── Load model ────────────────────────────────────────────────────────────
st.markdown("#### Load Model for Inference")
ids = [e["id"] for e in reg]
sel = st.selectbox("Select model", ids,
                   format_func=lambda mid: next(
                       (f"{e['algo']} | acc={e['accuracy']*100:.1f}% | {e['tag']}"
                       for e in reg if e['id']==mid), mid))
if st.button("📂 Load Selected Model as Active", use_container_width=True):
    entry = next(e for e in reg if e["id"]==sel)
    st.session_state.train_result = {
        "pipeline": entry["pipeline"], "label_encoder": entry["le"],
        "classes":  entry["classes"],  "algo": entry["algo"],
        "accuracy": entry["accuracy"], "cv_acc": __import__("numpy").array([entry["cv_mean"]]),
        "cv_f1":    __import__("numpy").array([entry["f1_macro"]]),
        "report":   {}, "cm": __import__("numpy").zeros((1,1)),
        "pred_df":  pd.DataFrame(), "train_n": entry["train_n"],
        "test_n": 0, "train_time": 0,
    }
    st.success(f"✅ `{sel}` loaded as active model. Go to 🔮 Live Testing.")

st.markdown("---")
# ── Download ──────────────────────────────────────────────────────────────
st.markdown("#### Download Model")
dl_sel = st.selectbox("Model to download", ids, key="dl_sel")
dl_entry = next(e for e in reg if e["id"]==dl_sel)
buf = io.BytesIO()
joblib.dump({"pipeline": dl_entry["pipeline"], "label_encoder": dl_entry["le"],
             "classes": dl_entry["classes"], "algo": dl_entry["algo"]}, buf)
buf.seek(0)
st.download_button(f"⬇️ Download {dl_sel}.joblib", data=buf,
                   file_name=f"{dl_sel}.joblib", use_container_width=True)

st.markdown("---")
# ── Delete ────────────────────────────────────────────────────────────────
st.markdown("#### Delete Model")
del_sel = st.selectbox("Model to delete", ids, key="del_sel")
if st.button("🗑️ Delete", use_container_width=True):
    st.session_state.registry = [e for e in reg if e["id"] != del_sel]
    st.success(f"Deleted `{del_sel}`"); st.rerun()
