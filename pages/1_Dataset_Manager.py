"""BotTrainer — Page 1: Dataset Manager"""
import json, sys, pandas as pd, streamlit as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.nlu_engine import dataset_stats
from utils.charts import intent_dist

st.set_page_config(page_title="Dataset Manager", page_icon="📁", layout="wide")
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#0E1117;}
[data-testid="stSidebar"]{background:#0A0F1A;border-right:1px solid #1E2A3A;}
[data-testid="metric-container"]{background:#131C2E;border:1px solid #1E2A3A;border-radius:10px;padding:16px!important;}
h1{color:#00C9A7!important;}h2,h3{color:#E8ECF4!important;}
</style>""", unsafe_allow_html=True)

if "dataset" not in st.session_state:
    st.session_state.dataset = None

st.title("📁 Dataset Manager")
st.markdown("Upload, explore, annotate and manage your NLU training data.")
st.markdown("---")

# ── Load ──────────────────────────────────────────────────────────────────────
col_up, col_sample = st.columns([2,1])
with col_up:
    st.markdown("#### Upload Dataset")
    f = st.file_uploader("JSON or CSV", type=["json","csv"],
                         help="JSON: list of {text,intent}  |  CSV: columns text,intent")
    if f:
        try:
            df = pd.DataFrame(json.load(f)) if f.name.endswith(".json") else pd.read_csv(f)
            if "text" not in df.columns or "intent" not in df.columns:
                st.error("Need 'text' and 'intent' columns.")
            else:
                st.session_state.dataset = df[["text","intent"]].dropna().reset_index(drop=True)
                st.success(f"✅ Loaded {len(df)} samples · {df['intent'].nunique()} intents")
        except Exception as e:
            st.error(f"Error: {e}")

with col_sample:
    st.markdown("#### Built-in Sample")
    if st.button("📦 Load Sample Dataset", use_container_width=True, type="primary"):
        p = Path(__file__).parent.parent / "data" / "sample_dataset.json"
        data = json.loads(p.read_text())
        st.session_state.dataset = pd.DataFrame(data)
        st.success("Sample dataset loaded!")
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.dataset = None
        st.rerun()

# ── Explore ───────────────────────────────────────────────────────────────────
if st.session_state.dataset is not None:
    df = st.session_state.dataset
    s  = dataset_stats(df)
    st.markdown("---")
    st.markdown("#### Overview")
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Total Samples",  s["total"])
    m2.metric("Unique Intents", s["num_intents"])
    m3.metric("Avg / Intent",   f"{s['avg_per_intent']:.1f}")
    m4.metric("Avg Words",      f"{s['avg_words']:.1f}")
    m5.metric("Imbalance Ratio",f"{s['imbalance']:.2f}")

    st.markdown("---")
    ca, cb = st.columns([2,1])
    with ca:
        st.plotly_chart(intent_dist(s["counts"]), use_container_width=True)
    with cb:
        st.markdown("#### Intent Registry")
        ic_df = pd.DataFrame([{"Intent":k,"Samples":v,"%":f"{v/s['total']*100:.1f}%"}
                               for k,v in s["counts"].items()]).sort_values("Samples",ascending=False)
        st.dataframe(ic_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Browse Samples")
    fc1,fc2,fc3 = st.columns([2,2,3])
    with fc1: intent_f = st.selectbox("Filter Intent", ["All"]+sorted(df["intent"].unique().tolist()))
    with fc2: sort_f   = st.selectbox("Sort", ["Default","Intent A-Z","Text Length"])
    with fc3: search   = st.text_input("🔎 Search", placeholder="Search utterances...")

    view = df.copy()
    if intent_f != "All": view = view[view.intent == intent_f]
    if search: view = view[view.text.str.contains(search, case=False, na=False)]
    if sort_f == "Intent A-Z": view = view.sort_values("intent")
    elif sort_f == "Text Length": view = view.assign(_l=view.text.str.len()).sort_values("_l",ascending=False).drop("_l",axis=1)
    st.caption(f"Showing {len(view)} of {len(df)} samples")
    st.dataframe(view.reset_index(drop=True), use_container_width=True, height=320, hide_index=True)

    st.markdown("---")
    st.markdown("#### Add New Sample")
    ac1,ac2,ac3 = st.columns([3,2,1])
    with ac1: new_text   = st.text_input("Utterance")
    with ac2: new_intent = st.selectbox("Intent", ["-- new --"]+sorted(df.intent.unique().tolist()), key="ni_sel")
    with ac3: custom_intent = st.text_input("New intent name", key="ni_custom",
                                             disabled=(new_intent != "-- new --"))
    if st.button("➕ Add Sample", type="primary"):
        t = new_text.strip()
        i = custom_intent.strip() if new_intent == "-- new --" else new_intent
        if t and i:
            st.session_state.dataset = pd.concat(
                [df, pd.DataFrame([{"text":t,"intent":i}])], ignore_index=True)
            st.success(f"Added: '{t}' → {i}"); st.rerun()
        else:
            st.warning("Fill in both text and intent.")

    st.markdown("---")
    st.markdown("#### Export")
    ec1,ec2 = st.columns(2)
    with ec1:
        st.download_button("⬇️ JSON", json.dumps(df[["text","intent"]].to_dict("records"),indent=2),
                           "nlu_dataset.json", "application/json", use_container_width=True)
    with ec2:
        st.download_button("⬇️ CSV", df[["text","intent"]].to_csv(index=False),
                           "nlu_dataset.csv", "text/csv", use_container_width=True)
else:
    st.info("👆 Upload a dataset or click Load Sample Dataset to begin.")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("**JSON format**")
        st.code('[{"text":"book a flight","intent":"book_flight"},...]', language="json")
    with c2:
        st.markdown("**CSV format**")
        st.code("text,intent\nbook a flight,book_flight\n...", language="text")
