import streamlit as st
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
import io
import speech_recognition as sr
from src.nlu_pipeline import predict, predict_from_image
from src.evaluator import evaluate

# ─────────────────────────────────────────
#  PAGE CONFIG & GLOBAL STYLES
# ─────────────────────────────────────────
st.set_page_config(
    page_title="𝓑𝓸𝓽𝓣𝓻𝓪𝓲𝓷𝓮𝓻 · 𝓝𝓛𝓤 𝓢𝓽𝓾𝓭𝓲𝓸",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide Streamlit auto page navigation
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display: none !important;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f0f1a !important;
    border-right: 1px solid #1e1e35;
}

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p {
    color: #c0c0d8 !important;
}

/* ── Headings ── */
h1 { 
    background: linear-gradient(135deg, #7c6fff, #ff6fd8, #6fffa8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    font-size: 3.4rem !important;
    letter-spacing: -0.03em;
}

h2, h3 {
    color: #c0c0d8 !important;
    font-weight: 600 !important;
}

/* ── Cards ── */
.card {
    background: #13131f;
    border: 1px solid #1e1e35;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}

.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #7c6fff, #ff6fd8, #6fffa8);
}

/* ── Mode Tabs ── */
.mode-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6666aa;
    margin-bottom: 8px;
}

/* ── Metric Cards ── */
.metric-card {
    background: #0f0f1a;
    border: 1px solid #1e1e35;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(135deg, #7c6fff, #ff6fd8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-label {
    font-size: 0.78rem;
    color: #6666aa;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}

/* ── Badge ── */
.intent-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 100px;
    background: linear-gradient(135deg, #7c6fff22, #ff6fd822);
    border: 1px solid #7c6fff55;
    color: #b0a8ff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    font-weight: 600;
}

/* ── Input overrides ── */
.stTextInput input, .stTextArea textarea {
    background: #0f0f1a !important;
    border: 1px solid #2a2a45 !important;
    border-radius: 10px !important;
    color: #e8e8f0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #7c6fff !important;
    box-shadow: 0 0 0 2px #7c6fff22 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #7c6fff, #ff6fd8) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: 0.03em;
    padding: 10px 28px !important;
    transition: opacity 0.2s !important;
}

.stButton > button:hover {
    opacity: 0.85 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0f0f1a;
    border-radius: 12px;
    padding: 4px;
    gap: 12px;
    border: 1px solid #1e1e35;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #6666aa !important;
    border-radius: 8px !important;
    font-weight: 700;
    border: none !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7c6fff33, #ff6fd833) !important;
    color: #c0b0ff !important;
    border: 1px solid #7c6fff44 !important;
}

/* ── Code / JSON ── */
.stCode, code {
    background: #0a0a14 !important;
    border: 1px solid #1e1e35 !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: #6fffa8 !important;
}

/* ── Alerts ── */
.stSuccess {
    background: #0d1f15 !important;
    border: 1px solid #1a4a2a !important;
    color: #6fffa8 !important;
    border-radius: 10px !important;
}

.stError {
    background: #1f0d0d !important;
    border: 1px solid #4a1a1a !important;
    border-radius: 10px !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #0f0f1a !important;
    border: 1px dashed #2a2a45 !important;
    border-radius: 12px !important;
}

/* ── Divider ── */
hr {
    border-color: #1e1e35 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #2a2a45; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 30px 0 20px;'>
        <div style='font-size:4.5rem;'>🤖</div>
        <div style='font-size:2.3rem; font-weight:700; 
             background:linear-gradient(135deg,#7c6fff,#ff6fd8);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            𝓑𝓸𝓽𝓣𝓻𝓪𝓲𝓷𝓮𝓻 
        </div>
        <div style='font-size:0.75rem; color:#6666aa; letter-spacing:0.15em; 
             text-transform:uppercase; margin-top:2px;'>NLU Studio</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    menu = st.selectbox(
        "Navigation",
        ["Single Prediction", "Evaluate Model"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='color:#6666aa; font-size:0.88rem; line-height:1.8;'>
    <b style='color:#c0c0ff; font-size:1.2rem;'>⚡ 𝓘𝓷𝓹𝓾𝓽 𝓜𝓸𝓭𝓮𝓼 </b><br>
    ✏️ Text — type your query<br>
    🎙️ Voice — record audio<br>
    🖼️ Image — upload screenshot<br><br>
    <b style='color:#9090c0;'>Powered by</b><br>
    Ollama · llama3 + llava
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────
st.markdown("<h1>🤖 𝓑𝓸𝓽𝓣𝓻𝓪𝓲𝓷𝓮𝓻 · 𝓝𝓛𝓤 𝓢𝓽𝓾𝓭𝓲𝓸🚀</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#6666aa; margin-top:-12px; margin-bottom:28px;'>"
    "Intent classification & entity extraction powered by Ollama (local LLM)</p>",
    unsafe_allow_html=True
)


# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────
def render_result(result: dict, extracted_text: str = ""):
    intent = result.get("intent", "unknown")
    confidence = result.get("confidence", 0.0)
    entities = result.get("entities", {})

    st.success("Prediction Complete")

    if extracted_text:
        st.markdown(f"""
        <div class='card' style='margin-bottom:16px;'>
            <div class='mode-label'>Extracted Text from Image</div>
            <div style='color:#c0c0d8; font-family:"JetBrains Mono",monospace;
                 background:#0a0a14; padding:12px; border-radius:8px; margin-top:8px;'>
                {extracted_text}
            </div>
        </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        bar_width = int(confidence * 100)
        bar_color = "#6fffa8" if confidence >= 0.8 else "#ffcc6f" if confidence >= 0.5 else "#ff6f6f"
        st.markdown(f"""
        <div class='card'>
            <div class='mode-label'>Intent Detected</div>
            <div style='margin-top:8px;'>
                <span class='intent-badge'>{intent}</span>
            </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        bar_width = int(confidence * 100)
        bar_color = "#6fffa8" if confidence >= 0.8 else "#ffcc6f" if confidence >= 0.5 else "#ff6f6f"
        st.markdown(f"""
        <div class='card'>
            <div class='mode-label'>Confidence Score</div>
            <div class='metric-value' style='margin-top:8px;'>{round(confidence*100, 1)}%</div>
            <div style='background:#1a1a2e; border-radius:100px; height:6px; margin-top:12px;'>
                <div style='background:{bar_color}; width:{bar_width}%; 
                     height:6px; border-radius:100px; 
                     transition:width 0.5s ease;'></div>
            </div>
        </div>""", unsafe_allow_html=True)

    if entities:
        st.markdown("<div class='mode-label' style='margin-top:8px;'>Entities</div>", unsafe_allow_html=True)
        st.code(json.dumps(entities, indent=2), language="json")


# ─────────────────────────────────────────
#  SINGLE PREDICTION PAGE
# ─────────────────────────────────────────
if "Single" in menu:

    if "current_text" not in st.session_state:
        st.session_state.current_text = ""

    # Input mode tabs
    tab_text, tab_voice, tab_image = st.tabs(["Text", "Voice", "Image"])

    # ── TEXT TAB ──
    with tab_text:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='mode-label'>Type your message</div>", unsafe_allow_html=True)
        user_input = st.text_input(
            "Message",
            value=st.session_state.current_text,
            placeholder="e.g. Book a flight to Mumbai for tomorrow...",
            label_visibility="collapsed",
        )

        col_btn, _ = st.columns([1, 3])
        with col_btn:
            predict_text = st.button("Predict Intent", key="btn_text")

        st.markdown("</div>", unsafe_allow_html=True)

        if predict_text:
            if user_input.strip():
                with st.spinner("Analyzing intent..."):
                    result = predict(user_input)
                render_result(result)
            else:
                st.warning("Please enter a message first.")

    # ── VOICE TAB ──
    with tab_voice:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='mode-label'>Record your voice</div>", unsafe_allow_html=True)

        audio_value = st.audio_input("Click the mic to record", label_visibility="collapsed")

        if audio_value:
            st.audio(audio_value)
            with st.spinner("Transcribing audio..."):
                try:
                    r = sr.Recognizer()
                    with sr.AudioFile(audio_value) as source:
                        audio_data = r.record(source)
                        transcribed = r.recognize_google(audio_data)

                    st.session_state.current_text = transcribed
                    st.markdown(f"""
                    <div style='background:#0f0f1a; border:1px solid #1e1e35; border-radius:10px;
                         padding:12px 16px; margin:12px 0; color:#c0c0d8;'>
                        🎤 <b>Transcribed:</b> {transcribed}
                    </div>""", unsafe_allow_html=True)

                    with st.spinner("Predicting intent..."):
                        result = predict(transcribed)
                    render_result(result)

                except sr.UnknownValueError:
                    st.error("Could not understand audio. Please speak clearly and try again.")
                except sr.RequestError as e:
                    st.error(f"Speech recognition service error: {e}")
                except Exception as e:
                    st.error(f"Transcription failed: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # ── IMAGE TAB ──
    with tab_image:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='mode-label'>Upload an image (screenshot, photo, handwriting)</div>", unsafe_allow_html=True)

        uploaded_img = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            label_visibility="collapsed",
        )

        if uploaded_img:
            img_bytes = uploaded_img.read()
            img = Image.open(io.BytesIO(img_bytes))

            col_img, col_info = st.columns([1, 1])
            with col_img:
                st.image(img, caption="Uploaded Image", use_container_width=True)
            with col_info:
                st.markdown(f"""
                <div style='background:#0f0f1a; border:1px solid #1e1e35; 
                     border-radius:10px; padding:16px; color:#9090c0; font-size:0.85rem;'>
                    <b style='color:#c0c0d8;'>File Info</b><br><br>
                    📄 {uploaded_img.name}<br>
                    📐 {img.size[0]} × {img.size[1]} px<br>
                    🎨 {img.mode} mode<br>
                    💾 {len(img_bytes) // 1024} KB
                </div>""", unsafe_allow_html=True)

            # Determine media type
            ext = uploaded_img.name.lower().rsplit(".", 1)[-1]
            mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                        "png": "image/png", "webp": "image/webp", "bmp": "image/bmp"}
            media_type = mime_map.get(ext, "image/jpeg")

            col_btn2, _ = st.columns([1, 3])
            with col_btn2:
                predict_img = st.button("Analyze Image", key="btn_img")

            if predict_img:
                with st.spinner("Reading image and predicting intent..."):
                    result = predict_from_image(img_bytes, media_type)
                extracted = result.pop("extracted_text", "")
                render_result(result, extracted_text=extracted)

        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  EVALUATE MODEL PAGE
# ─────────────────────────────────────────
elif "Evaluate" in menu:
    st.markdown("### Model Evaluation")
    st.markdown(
        "<p style='color:#6666aa;'>Run predictions on sample test data and review metrics.</p>",
        unsafe_allow_html=True
    )

    # ── Custom test data upload OR use defaults ──
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='mode-label'>Test Dataset</div>", unsafe_allow_html=True)

    default_test_data = [
        {"text": "Book flight to Delhi", "intent": "book_flight"},
        {"text": "I want to travel to Mumbai tomorrow", "intent": "book_flight"},
        {"text": "Order pizza", "intent": "order_food"},
        {"text": "I want to eat biryani", "intent": "order_food"},
        {"text": "Weather in Pune", "intent": "check_weather"},
        {"text": "Is it raining in Mumbai?", "intent": "check_weather"},
        {"text": "Hello there", "intent": "greet"},
        {"text": "Bye, see you later", "intent": "goodbye"},
    ]

    uploaded_csv = st.file_uploader(
        "Upload custom test CSV (columns: text, intent) — or use defaults below",
        type=["csv"],
        label_visibility="visible",
    )

    if uploaded_csv:
        try:
            df_custom = pd.read_csv(uploaded_csv)
            if "text" in df_custom.columns and "intent" in df_custom.columns:
                test_data = df_custom[["text", "intent"]].dropna().to_dict("records")
                st.success(f"Loaded {len(test_data)} samples from CSV")
            else:
                st.error("CSV must have 'text' and 'intent' columns.")
                test_data = default_test_data
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            test_data = default_test_data
    else:
        test_data = default_test_data

    df_test = pd.DataFrame(test_data)
    st.dataframe(df_test, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Run Evaluation"):
        true_labels, predicted_labels = [], []
        progress = st.progress(0, text="Running predictions...")

        for i, item in enumerate(test_data):
            res = predict(item["text"])
            true_labels.append(item["intent"])
            predicted_labels.append(res["intent"])
            progress.progress((i + 1) / len(test_data), text=f"Processing {i+1}/{len(test_data)}...")

        progress.empty()
        results = evaluate(true_labels, predicted_labels)

        # ── Metrics ──
        st.markdown("### Results")
        col_acc, col_total, col_correct = st.columns(3)

        correct = sum(t == p for t, p in zip(true_labels, predicted_labels))

        with col_acc:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{results['accuracy']}%</div>
                <div class='metric-label'>Accuracy</div>
            </div>""", unsafe_allow_html=True)

        with col_total:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{len(test_data)}</div>
                <div class='metric-label'>Total Samples</div>
            </div>""", unsafe_allow_html=True)

        with col_correct:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{correct}</div>
                <div class='metric-label'>Correct Predictions</div>
            </div>""", unsafe_allow_html=True)

        # ── Prediction Table ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='mode-label'>Prediction Details</div>", unsafe_allow_html=True)
        df_results = pd.DataFrame({
            "Input Text": [d["text"] for d in test_data],
            "True Intent": true_labels,
            "Predicted Intent": predicted_labels,
            "Match": ["✅" if t == p else "❌" for t, p in zip(true_labels, predicted_labels)],
        })
        st.dataframe(df_results, use_container_width=True, hide_index=True)

        # ── Confusion Matrix ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='mode-label'>Confusion Matrix</div>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor("#0f0f1a")
        ax.set_facecolor("#0f0f1a")

        sns.heatmap(
            results["confusion_matrix"],
            annot=True,
            fmt="d",
            ax=ax,
            xticklabels=results["labels"],
            yticklabels=results["labels"],
            cmap="PuRd",
            linewidths=0.5,
            linecolor="#1e1e35",
            cbar_kws={"shrink": 0.8},
        )

        ax.set_xlabel("Predicted", color="#9090c0", fontsize=11)
        ax.set_ylabel("True", color="#9090c0", fontsize=11)
        ax.tick_params(colors="#9090c0", labelsize=9)
        plt.xticks(rotation=30, ha="right")

        st.pyplot(fig)
        plt.close()

        # ── Per-class Report ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='mode-label'>Per-class Report</div>", unsafe_allow_html=True)
        report_df = pd.DataFrame(results["report"]).T.round(2)
        report_df = report_df[report_df.index.isin(results["labels"])]
        st.dataframe(report_df, use_container_width=True)
# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer" style="
        text-align:center;
        padding:20px 0;
        font-size:1rem;
        color:#8888cc;
        border-top:1px solid #1e1e35;
        margin-top:40px;
        letter-spacing:0.05em;
    ">
        &copy; 2026 &nbsp;|&nbsp; 
        <b style="color:#c0c0ff;">𝕾𝖚𝖗𝖞𝖆 𝕺𝖒𝖆𝖗</b> &nbsp;|&nbsp;
        🏢 <span style="color:#aaaaff;">
        𝓘𝓷𝓯𝓸𝓼𝔂𝓼 𝓼𝓹𝓻𝓲𝓷𝓰𝓫𝓸𝓪𝓻𝓭
        </span> <br>
        🧠 <span style="
            background:linear-gradient(135deg,#7c6fff,#ff6fd8);
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;
            font-weight:600;
        ">
        BotTrainer · NLU Studio 🚀
        </span> &nbsp;|&nbsp;
    </div>
    """,
    unsafe_allow_html=True,
)