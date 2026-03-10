"""BotTrainer — Page 4: Live Testing"""
import sys, pandas as pd, streamlit as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.nlu_engine import predict, extract_entities
from utils.charts import all_intent_scores

st.set_page_config(page_title="Live Testing", page_icon="🔮", layout="wide")
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#0E1117;}
[data-testid="stSidebar"]{background:#0A0F1A;border-right:1px solid #1E2A3A;}
[data-testid="metric-container"]{background:#131C2E;border:1px solid #1E2A3A;border-radius:10px;padding:16px!important;}
h1{color:#00C9A7!important;}h2,h3{color:#E8ECF4!important;}
</style>""", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
if "history"        not in st.session_state: st.session_state.history       = []
if "pending_input"  not in st.session_state: st.session_state.pending_input = ""
if "last_result"    not in st.session_state: st.session_state.last_result   = None

st.title("🔮 Live Testing")
st.markdown("Type any message to get real-time intent predictions and entity extraction.")
st.markdown("---")

if "train_result" not in st.session_state or not st.session_state.train_result:
    st.warning("⚠️ No model trained. Go to **⚡ Trainer** first.")
    st.stop()

res  = st.session_state.train_result
pipe = res["pipeline"]
le   = res["label_encoder"]

st.success(f"✅ Model: `{res['algo']}` · Accuracy `{res['accuracy']*100:.1f}%` · {len(res['classes'])} intents")
st.markdown("---")

# ── Quick sample buttons (ABOVE the text area, set pending_input then rerun) ──
st.markdown("#### ⚡ Quick Test Samples")
SAMPLES = [
    "will it rain tomorrow in london",
    "book a flight to paris on monday",
    "set an alarm for 7 30 am",
    "play some jazz music please",
    "order two large pepperoni pizzas",
    "call my dentist at 555 1234",
    "navigate to the nearest hospital",
    "how far is the airport",
]
btn_cols = st.columns(4)
for i, s in enumerate(SAMPLES):
    if btn_cols[i % 4].button(s, key=f"qs_{i}", use_container_width=True):
        st.session_state.pending_input = s
        st.rerun()

st.markdown("---")

# ── Text input — value driven by pending_input ────────────────────────────────
st.markdown("#### 💬 Message Input")
col_in, col_btn = st.columns([5, 1])
with col_in:
    # Use pending_input as the default value; widget key is NOT used for quick-fill
    user_text = st.text_area(
        "Enter a message:",
        value=st.session_state.pending_input,
        height=90,
        placeholder="e.g. book a flight to london tomorrow",
    )
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 PREDICT", type="primary", use_container_width=True)

# Clear pending after it has been consumed by the text area
if st.session_state.pending_input:
    st.session_state.pending_input = ""

# ── Run prediction ────────────────────────────────────────────────────────────
if predict_btn:
    if user_text and user_text.strip():
        r = predict(pipe, le, user_text.strip())
        st.session_state.last_result = r
        st.session_state.history.insert(0, r)
    else:
        st.warning("Please enter some text first.")

# ── Show result ───────────────────────────────────────────────────────────────
r = st.session_state.last_result
if r:
    st.markdown("---")
    st.markdown("#### 🎯 Prediction Result")

    conf  = r["conf"]
    level = "🟢 HIGH" if conf > 0.7 else "🟡 MEDIUM" if conf > 0.4 else "🔴 LOW"

    k1, k2, k3 = st.columns(3)
    k1.metric("Intent",           r["intent"])
    k2.metric("Confidence",       f"{conf*100:.1f}%")
    k3.metric("Confidence Level", level)

    st.markdown("---")
    cl, cr = st.columns(2)

    with cl:
        if r["all_intents"]:
            st.markdown("**All Intent Scores**")
            st.plotly_chart(all_intent_scores(r["all_intents"]), use_container_width=True)

    with cr:
        st.markdown("**Extracted Entities**")
        ents = r["entities"]
        if ents:
            for e in ents:
                st.markdown(
                    f"`{e['type']}`  **{e['value']}**  "
                    f"<span style='color:#475569;font-size:11px'>pos {e['start']}–{e['end']}</span>",
                    unsafe_allow_html=True,
                )
            st.dataframe(
                pd.DataFrame(ents)[["type", "value", "start", "end"]],
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No entities detected in this input.")

        st.markdown("---")
        st.markdown("**Preprocessed Text**")
        st.code(f"Input : {r['text']}\nClean : {r['clean']}", language="text")

# ── Inference history ──────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown("#### 🕘 Inference History")
    hc1, hc2 = st.columns([5, 1])
    with hc2:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history     = []
            st.session_state.last_result = None
            st.rerun()

    hist_rows = [
        {
            "Text":       h["text"],
            "Intent":     h["intent"],
            "Confidence": f"{h['conf']*100:.1f}%",
            "Entities":   ", ".join(f"{e['value']} ({e['type']})" for e in h["entities"]) or "—",
        }
        for h in st.session_state.history[:25]
    ]
    st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, height=260, hide_index=True)
    st.download_button(
        "⬇️ Download History (CSV)",
        pd.DataFrame(hist_rows).to_csv(index=False),
        "inference_history.csv", "text/csv",
    )

# ── Entity legend ──────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📖 Supported Entity Types"):
    st.markdown("""
| Type | Examples |
|---|---|
| DATE | today, tomorrow, next monday, 12/25 |
| TIME | 7am, 3:30pm, midnight |
| DURATION | 30 minutes, 2 hours |
| NUMBER | 2, 42, 3.14 |
| PHONE | 555-1234, +1 (800) 555-0100 |
| EMAIL | user@example.com |
| CURRENCY | $50, 20 dollars |
| LOCATION | London, Paris, Tokyo, New York… |
""")