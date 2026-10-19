import csv
import html
import io
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

# ------------------------------------------------------------
# Project setup and existing model inference
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.predict import predict

st.set_page_config(
    page_title="PalmSense | FFB Ripeness",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
defaults = {
    "nav_page": "Home",
    "history": [],
    "current_result": None,
    "current_image": None,
    "current_filename": None,
    "current_time": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ------------------------------------------------------------
# Attractive traditional website styling
# ------------------------------------------------------------
st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root {
  --green-950: #073b2d;
  --green-900: #0a4937;
  --green-800: #116347;
  --green-700: #168052;
  --green-600: #1d9a62;
  --green-100: #e6f6ed;
  --ink: #18352c;
  --muted: #65776f;
  --line: #e1ebe5;
  --surface: #ffffff;
  --page: #f5f8f5;
}
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
  font-family: 'DM Sans', sans-serif;
}
.stApp {
  background:
    radial-gradient(circle at 88% 2%, rgba(51, 168, 104, .08), transparent 22rem),
    var(--page);
  color: var(--ink);
}
[data-testid="stHeader"] { background: rgba(245,248,245,.92); }
[data-testid="stToolbar"] { right: 1rem; }
.main .block-container {
  max-width: 1320px;
  padding-top: 1.15rem;
  padding-bottom: 3.5rem;
}
section[data-testid="stSidebar"] { display: none; }
.brand-bar {
  display:flex; align-items:center; justify-content:space-between;
  gap:18px; padding: 16px 22px; margin: 0 0 24px;
  background: rgba(255,255,255,.94); border:1px solid var(--line);
  border-radius:18px; box-shadow:0 8px 28px rgba(21,65,43,.055);
}
.brand-left { display:flex; align-items:center; gap:12px; }
.brand-mark {
  display:flex; align-items:center; justify-content:center;
  width:48px; height:48px; border-radius:15px;
  background:linear-gradient(145deg,#1e9b63,#0b6548);
  color:white; font-size:27px; box-shadow:0 7px 16px rgba(20,130,79,.22);
}
.brand-name { color:var(--green-950); font:800 27px 'Manrope',sans-serif; letter-spacing:-.6px; }
.brand-tagline { color:var(--muted); font-size:15px; margin-top:3px; }
.status-pill {
  padding:8px 13px; border-radius:999px; color:#147347;
  background:#e8f7ee; border:1px solid #cbead7; font-size:12px; font-weight:700;
}
div[role="radiogroup"] {
  gap:8px; padding:8px; border:1px solid var(--line); border-radius:14px;
  background:white; margin-bottom:26px;
}
div[role="radiogroup"] label {
  border-radius:10px; padding:8px 17px; font-weight:700; color:#496258;
}
div[role="radiogroup"] label:has(input:checked) {
  background:#e5f5eb; color:#0b6a45;
}
.hero {
  position:relative; overflow:hidden; border-radius:26px;
  padding:48px 46px; margin-bottom:28px; color:white;
  background:
    radial-gradient(circle at 84% 16%, rgba(166,242,185,.22), transparent 14rem),
    radial-gradient(circle at 95% 100%, rgba(255,211,125,.17), transparent 17rem),
    linear-gradient(115deg,#073d2e 0%,#0c6244 55%,#168154 100%);
  box-shadow:0 18px 42px rgba(7,73,48,.16);
}
.hero:after {
  content:"🌴"; position:absolute; right:30px; bottom:-48px;
  font-size:190px; opacity:.14; transform:rotate(-7deg);
}
.hero-content { position:relative; z-index:1; max-width:790px; }
.eyebrow {
  display:inline-block; padding:7px 12px; margin-bottom:16px;
  border:1px solid rgba(255,255,255,.24); border-radius:999px;
  background:rgba(255,255,255,.10); color:#d6ffe3;
  font-size:14px; font-weight:800; letter-spacing:1px; text-transform:uppercase;
}
.hero h1 { color:white; font:800 clamp(32px,4vw,51px)/1.12 'Manrope',sans-serif; margin:0 0 16px; letter-spacing:-1.5px; }
.hero p { color:rgba(255,255,255,.9); font-size:21px; line-height:1.7; max-width:700px; margin:0 0 24px; }
.hero-points { display:flex; flex-wrap:wrap; gap:12px 23px; font-size:16px; font-weight:700; color:#f2fff5; }
.section-heading { color:var(--green-950); font:800 33px 'Manrope',sans-serif; letter-spacing:-.7px; margin:20px 0 6px; }
.section-sub { color:var(--muted); font-size:19px; line-height:1.65; margin:0 0 22px; }
.feature-card, .content-card, .metric-card, .history-card {
  background:var(--surface); border:1px solid var(--line); border-radius:18px;
  box-shadow:0 7px 24px rgba(20,64,43,.045);
}
.feature-card { padding:23px; min-height:205px; }
.feature-icon {
  display:flex; align-items:center; justify-content:center; width:48px; height:48px;
  border-radius:14px; background:#e8f7ee; font-size:24px; margin-bottom:16px;
}
.feature-card h3 { color:var(--green-950); font:800 22px 'Manrope',sans-serif; margin:0 0 9px; }
.feature-card p { color:var(--muted); font-size:17px; line-height:1.65; margin:0; }
.content-card { padding:25px; margin:14px 0 22px; }
.content-title { color:var(--green-950); font:800 24px 'Manrope',sans-serif; margin:0 0 7px; }
.content-subtitle { color:var(--muted); font-size:18px; line-height:1.6; margin:0 0 18px; }
.step-row { display:flex; align-items:center; gap:14px; margin:14px 0; }
.step-num { width:36px; height:36px; flex:0 0 36px; border-radius:50%; display:flex; align-items:center; justify-content:center; color:#fff; background:var(--green-700); font-weight:800; }
.step-text strong { color:var(--green-950); display:block; font-size:18px; }
.step-text span { color:var(--muted); font-size:16px; }
div.stButton > button, div.stDownloadButton > button {
  border:0; border-radius:12px; min-height:47px; padding:0 20px;
  font-weight:800; color:white; background:linear-gradient(135deg,#16935c,#0a6948);
  box-shadow:0 7px 17px rgba(15,126,75,.15); transition:transform .15s, box-shadow .15s;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
  color:white; transform:translateY(-1px); box-shadow:0 10px 22px rgba(15,126,75,.22);
  border:0; background:linear-gradient(135deg,#20a76b,#0d7851);
}

/* Readable application text and controls */
.stApp, .stApp p, .stApp label, .stApp li, .stApp [data-testid="stCaptionContainer"] {
  font-size: 18px;
  line-height: 1.65;
}
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"],
.stFileUploader section, .stRadio label, .stButton button, .stDownloadButton button {
  font-size: 17px !important;
}
div[role="radiogroup"] label {
  font-size: 18px;
  padding: 12px 22px;
}

[data-testid="stFileUploaderDropzone"] {
  background:#fafffc; border:2px dashed #91c9ad; border-radius:15px;
}
[data-testid="stFileUploaderDropzone"] button {
  border-radius:9px; font-weight:700;
}
.metric-card { padding:22px; min-height:148px; }
.metric-label { color:#52685d; font-size:15px; font-weight:800; letter-spacing:.7px; text-transform:uppercase; margin-bottom:13px; }
.metric-value { color:var(--green-950); font:800 clamp(23px,2.5vw,32px) 'Manrope',sans-serif; overflow-wrap:anywhere; }
.metric-note { color:#65776f; font-size:15px; margin-top:9px; line-height:1.5; }
.result-image-title { color:var(--green-950); font:800 18px 'Manrope',sans-serif; margin-bottom:12px; }
.prob-row { margin:15px 0; }
.prob-head { display:flex; justify-content:space-between; gap:12px; color:#345449; font-size:17px; font-weight:700; margin-bottom:7px; }
.history-card { padding:19px 22px; margin-bottom:14px; }
.history-name { color:var(--green-950); font:800 18px 'Manrope',sans-serif; margin-bottom:7px; }
.history-meta { color:var(--muted); font-size:17px; line-height:1.8; }
.notice {
  background:#fff8e8; border:1px solid #f2dfb2; color:#75591b;
  border-radius:12px; padding:16px 18px; font-size:16px; line-height:1.7;
}
.footer { border-top:1px solid var(--line); text-align:center; color:#74847b; font-size:13px; padding:24px 0 4px; margin-top:45px; }
.footer strong { color:var(--green-700); }
@media(max-width:760px) {
  .main .block-container { padding-left:1rem; padding-right:1rem; }
  .brand-bar { padding:13px; }
  .brand-name { font-size:18px; }
  .brand-tagline { font-size:11px; }
  .status-pill { display:none; }
  .hero { padding:32px 24px; }
  .hero:after { font-size:105px; right:4px; }
  .hero p { font-size:16px; }
  .section-heading { font-size:25px; }
}
</style>
""")

# Header and traditional top navigation
st.html("""
<div class="brand-bar">
  <div class="brand-left">
    <div class="brand-mark">🌴</div>
    <div>
      <div class="brand-name">PalmSense</div>
      <div class="brand-tagline">Oil Palm Fruit Ripeness Intelligence</div>
    </div>
  </div>
  <div class="status-pill">● AI ANALYSIS PLATFORM</div>
</div>
""")

pages = ["Home", "Prediction", "History", "About"]
current_page = st.radio(
    "Main navigation",
    pages,
    horizontal=True,
    label_visibility="collapsed",
    key="nav_page",
)


def section_title(title, subtitle=""):
    st.markdown(f'<div class="section-heading">{html.escape(title)}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def show_hero():
    st.html("""
    <section class="hero">
      <div class="hero-content">
        <div class="eyebrow">Smart agriculture · Deep learning</div>
        <h1>Understand your oil palm fruit bunches with confidence.</h1>
        <p>Upload a Fresh Fruit Bunch (FFB) image to explore its predicted ripeness, model confidence, estimated bunch count, and visual mask result.</p>
        <div class="hero-points">
          <span>🌿 Four-class ripeness analysis</span>
          <span>🎯 Visual mask output</span>
          <span>📊 Count estimate</span>
        </div>
      </div>
    </section>
    """)


def get_sample_image():
    candidates = [
        PROJECT_ROOT / "dataset" / "raw" / "test" / "images" / "Empty_Bunch_020_jpg.rf.c1604df364f52ae4396cfbd563702d76.jpg",
        Path(r"D:\projects\oil\DS\FFB_Ripeness_Extracted\FFB_Ripeness_Dataset\test\images\Empty_Bunch_020_jpg.rf.c1604df364f52ae4396cfbd563702d76.jpg"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Fallback: find the first image in the project's existing FFB dataset.
    roots = [
        PROJECT_ROOT / "dataset" / "raw",
        Path(r"D:\projects\oil\DS\FFB_Ripeness_Extracted\FFB_Ripeness_Dataset"),
    ]
    for root in roots:
        if root.exists():
            for split in ("test", "valid", "train"):
                image_dir = root / split / "images"
                if image_dir.exists():
                    found = next((p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}), None)
                    if found:
                        return found
    return None


def save_uploaded_image(uploaded_file):
    upload_dir = PROJECT_ROOT / "inference" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = Path(uploaded_file.name).name
    destination = upload_dir / f"{timestamp}_{safe_name}"
    image = Image.open(uploaded_file).convert("RGB")
    image.save(destination, format="JPEG", quality=95)
    return destination, image


def run_prediction(uploaded_file):
    try:
        with st.spinner("Analyzing your image with the trained model…"):
            image_path, image = save_uploaded_image(uploaded_file)
            result = predict(image_path)
        st.session_state.current_result = result
        st.session_state.current_image = image
        st.session_state.current_filename = uploaded_file.name
        st.session_state.current_time = datetime.now().strftime("%d %b %Y, %I:%M %p")
        st.session_state.history.insert(0, {
            "filename": uploaded_file.name,
            "time": st.session_state.current_time,
            "ripeness": result["ripeness"],
            "confidence": float(result["confidence"]),
            "count": float(result["estimated_count"]),
            "output_path": str(result.get("output_path", "")),
            "probabilities": result.get("probabilities", {}),
        })
        st.success("Analysis completed successfully.")
        return True
    except Exception as error:
        st.error("The image could not be analyzed. Check the file and model setup, then try again.")
        with st.expander("Technical error details"):
            st.exception(error)
        return False


def show_result_metrics(result):
    ripeness = str(result.get("ripeness", "Unknown"))
    confidence = float(result.get("confidence", 0.0))
    count = float(result.get("estimated_count", 0.0))
    a, b, c = st.columns(3)
    with a:
        st.html(f"""<div class="metric-card">
          <div class="metric-label">🌿 Predicted ripeness</div>
          <div class="metric-value">{html.escape(ripeness)}</div>
          <div class="metric-note">Highest-probability ripeness class.</div>
        </div>""")
    with b:
        st.html(f"""<div class="metric-card">
          <div class="metric-label">🎯 Model confidence</div>
          <div class="metric-value">{confidence * 100:.2f}%</div>
          <div class="metric-note">Confidence reported by the classifier.</div>
        </div>""")
    with c:
        st.html(f"""<div class="metric-card">
          <div class="metric-label">📊 Estimated FFB count</div>
          <div class="metric-value">{count:.2f}</div>
          <div class="metric-note">Model count-head output; not a guaranteed physical count.</div>
        </div>""")


def show_prediction_result():
    result = st.session_state.current_result
    if result is None:
        return

    st.markdown("---")
    section_title("Analysis report", "Your latest model output is summarized below.")
    show_result_metrics(result)

    left, right = st.columns(2, gap="large")
    with left:
        st.html('<div class="content-card"><div class="result-image-title">🖼️ Uploaded FFB image</div></div>')
        if st.session_state.current_image is not None:
            st.image(st.session_state.current_image, width="stretch")
        st.caption(f"File: {st.session_state.current_filename or 'Uploaded image'}")
    with right:
        st.html('<div class="content-card"><div class="result-image-title">🎯 Predicted mask visualization</div></div>')
        output_path = Path(str(result.get("output_path", "")))
        if output_path.is_file():
            st.image(str(output_path), width="stretch")
            st.caption("Model-generated mask overlay and bounding-box visualization.")
        else:
            st.info("The inference script did not provide a saved visualization for this result.")

    section_title("Ripeness probability", "Compare the model's reported probability for each class.")
    probabilities = result.get("probabilities", {})
    if probabilities:
        for class_name, probability in probabilities.items():
            probability = max(0.0, min(1.0, float(probability)))
            row_left, row_right = st.columns([1.5, 4])
            with row_left:
                st.markdown(f"**{html.escape(str(class_name))}**")
            with row_right:
                st.progress(probability, text=f"{probability * 100:.2f}%")
    else:
        st.info("Class probabilities were not included in the inference output.")

    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["Image", "Prediction time", "Predicted ripeness", "Confidence", "Estimated FFB count"])
    writer.writerow([
        st.session_state.current_filename or "",
        st.session_state.current_time or "",
        result.get("ripeness", ""),
        f'{float(result.get("confidence", 0.0)) * 100:.2f}%',
        f'{float(result.get("estimated_count", 0.0)):.2f}',
    ])
    st.download_button(
        "⬇️ Download this prediction as CSV",
        data=csv_buffer.getvalue(),
        file_name="palmsense_prediction_report.csv",
        mime="text/csv",
        width="stretch",
    )
    st.html("""<div class="notice">Research note: the mask target was generated from existing bounding-box annotations, so this overlay is a box-derived pseudo-mask prediction, not a manually annotated polygon segmentation result. The available training images also contain one annotated bunch per image, so count estimates should be interpreted cautiously.</div>""")
    
if st.button("🔍 Analyze FFB image", width="stretch", key="prediction_analyze"):
    st.write("Button clicked! Starting prediction...")

    success = run_prediction(uploaded_file)

    if success:
        st.success("Prediction finished! Loading results...")
        st.rerun()
    else:
        st.error("Prediction failed. Check the technical error details above.")



def show_home():
    show_hero()

    section_title("What can you do here?", "Choose one section from the menu above. You do not need to learn technical terms.")

    cols = st.columns(3, gap="large")
    cards = [
        ("📷", "Predict ripeness", "Upload a fruit-bunch photo and see the model's result."),
        ("🕘", "View history", "Check the predictions made in this current session."),
        ("ℹ️", "Learn about the app", "See what the model does and understand its limits."),
    ]
    for col, (icon, title, description) in zip(cols, cards):
        with col:
            st.html(f"""<div class="feature-card">
              <div class="feature-icon">{icon}</div>
              <h3>{title}</h3><p>{description}</p>
            </div>""")

    st.markdown("")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.html("""<div class="content-card">
          <div class="content-title">Ready to check a fruit bunch?</div>
          <div class="content-subtitle">Start with one clear JPG, JPEG, or PNG image. The app will show the predicted ripeness and confidence.</div>
        </div>""")
        st.button(
    "Start prediction  →",
    width="stretch",
    key="home_start_prediction",
    on_click=lambda: setattr(st.session_state, "nav_page", "Prediction")
)
    with right:
        sample = get_sample_image()
        st.html('<div class="content-card"><div class="content-title">Example image</div><div class="content-subtitle">An image from your existing dataset, if the local dataset folder is available.</div></div>')
        if sample:
            st.image(str(sample), width="stretch")
        else:
            st.info("No local example image was found. You can still upload your own image from the Prediction page.")


def show_prediction_page():
    section_title("Predict fruit ripeness", "Upload one clear fruit-bunch photo, then select Analyze.")
    st.html("""
    <div class="content-card">
      <div class="content-title">How to use it</div>
      <div class="step-row"><div class="step-num">1</div><div class="step-text"><strong>Choose a photo</strong><span>Select a clear JPG, JPEG, or PNG image.</span></div></div>
      <div class="step-row"><div class="step-num">2</div><div class="step-text"><strong>Check the preview</strong><span>Make sure the fruit bunch is visible.</span></div></div>
      <div class="step-row"><div class="step-num">3</div><div class="step-text"><strong>Select Analyze</strong><span>Wait for the result to appear below.</span></div></div>
    </div>
    """)
    uploaded_file = st.file_uploader(
        "Upload an FFB image",
        type=["jpg", "jpeg", "png"],
        help="Choose one clear oil palm Fresh Fruit Bunch image.",
        key="prediction_uploader",
    )
    if uploaded_file is not None:
        try:
            preview = Image.open(uploaded_file).convert("RGB")
            st.html('<div class="content-card"><div class="content-title">🖼️ Image preview</div><div class="content-subtitle">Confirm this is the image you want to analyze.</div></div>')
            st.image(preview, caption=uploaded_file.name, width="stretch")
            if st.button("🔍 Analyze FFB image", width="stretch", key="prediction_analyze"):
                run_prediction(uploaded_file)
                st.rerun()
        except Exception:
            st.error("This file could not be opened as an image. Please choose a valid JPG, JPEG, or PNG.")
    if st.session_state.current_result is not None:
        show_prediction_result()


def show_history():
    section_title("Prediction history", "Review results created during this app session.")
    if not st.session_state.history:
        st.html("""<div class="content-card">
          <div class="content-title">No predictions yet</div>
          <div class="content-subtitle">Your completed analyses will appear here. Open Prediction to upload your first image.</div>
        </div>""")
        if st.button("Go to prediction  →", width="stretch"):
            st.session_state.nav_page = "Prediction"
            st.rerun()
        return

    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["Image", "Time", "Ripeness", "Confidence (%)", "Estimated count"])
    for item in st.session_state.history:
        writer.writerow([item["filename"], item["time"], item["ripeness"], f'{item["confidence"] * 100:.2f}', f'{item["count"]:.2f}'])
    st.download_button("⬇️ Download full session history (CSV)", csv_buffer.getvalue(), "palmsense_history.csv", "text/csv")

    for index, item in enumerate(st.session_state.history, start=1):
        st.html(f"""<div class="history-card">
          <div class="history-name">🌴 {index}. {html.escape(str(item['ripeness']))}</div>
          <div class="history-meta"><b>Image:</b> {html.escape(str(item['filename']))}<br>
          <b>Time:</b> {html.escape(str(item['time']))}<br>
          <b>Confidence:</b> {float(item['confidence']) * 100:.2f}%<br>
          <b>Estimated count:</b> {float(item['count']):.2f}</div>
        </div>""")


def show_about():
    section_title("About PalmSense", "A simple tool that checks an oil palm fruit-bunch image using your trained model.")
    st.html("""<div class="content-card">
      <div class="content-title">What the application provides</div>
      <p>• Four-class ripeness classification: Empty Bunch, Overripe, Ripe, and Underripe.</p>
      <p>• A model-generated mask visualization.</p>
      <p>• An estimated FFB count value.</p>
      <p>• Class probabilities and downloadable CSV summaries.</p>
    </div>
    <div class="content-card">
      <div class="content-title">Model workflow</div>
      <div class="step-row"><div class="step-num">1</div><div class="step-text"><strong>Input image</strong><span>Resize and normalize for model input.</span></div></div>
      <div class="step-row"><div class="step-num">2</div><div class="step-text"><strong>Shared feature extractor</strong><span>ConvNeXt-FPN feature fusion with Transformer attention.</span></div></div>
      <div class="step-row"><div class="step-num">3</div><div class="step-text"><strong>Multi-task heads</strong><span>Mask output, ripeness class, and count estimate.</span></div></div>
    </div>
    <div class="notice">Model limitation: the mask training targets are rectangular pseudo-masks derived from the dataset's existing bounding boxes, not hand-drawn polygon masks. The existing FFB dataset has one annotated bunch per image, so it does not establish performance for dense multi-bunch counting.</div>""")
    st.markdown("")
    m1, m2, m3 = st.columns(3)
    m1.metric("Validation accuracy", "89.19%")
    m2.metric("Test accuracy", "82.14%")
    m3.metric("Best checkpoint epoch", "10")
    st.caption("These figures are from the existing training/evaluation run and are not a guarantee of accuracy on every new image.")


# ------------------------------------------------------------
# Page router
# ------------------------------------------------------------
if current_page == "Home":
    show_home()
elif current_page == "Prediction":
    show_prediction_page()
elif current_page == "History":
    show_history()
else:
    show_about()

st.html("""
<div class="footer">
  <strong>PalmSense</strong> · Oil Palm Fruit Ripeness Prediction<br>
  Built around the existing multi-task deep learning model · Smart agriculture 🌱
</div>
""")
