"""Heart Disease Risk Prediction System.  Run:  streamlit run app.py"""
import json, os, joblib, pandas as pd, plotly.graph_objects as go, streamlit as st
from train_model import COLS

st.set_page_config(page_title="Heart Risk Screening", page_icon="❤️", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("model/rf_heart.joblib") if os.path.exists("model/rf_heart.joblib") else None

model = load_model()
st.title("❤️ Heart Disease Risk Prediction System")
st.caption("Screening and decision support only — not a medical diagnosis.")
if model is None:
    st.error("Model not found. Run `python train_model.py` first."); st.stop()

CP = {1: "Typical angina", 2: "Atypical angina", 3: "Non-anginal pain", 4: "Asymptomatic"}
ECG = {0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"}
SLOPE = {1: "Upsloping", 2: "Flat", 3: "Downsloping"}
THAL = {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}
DEF = dict(age=50, sex=1, cp=1, trestbps=120, chol=200, fbs=0, restecg=0,
           thalach=150, exang=0, oldpeak=1.0, slope=1, ca=0, thal=3)
for k, v in DEF.items():
    st.session_state.setdefault(k, v)

# ---- Path 1: medical report image -> OpenCV -> OCR (fills the form below) ----
with st.expander("📄 Autofill from a medical report image (OpenCV + OCR)"):
    up = st.file_uploader("Upload report (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if up and st.button("Extract values"):
        try:
            from ocr_utils import run_ocr, extract_values
            text, processed = run_ocr(up.getvalue())
            found = extract_values(text)
            for k, v in found.items():
                st.session_state[k] = type(DEF[k])(v)
            c1, c2 = st.columns(2)
            c1.image(processed, caption="After preprocessing", use_container_width=True)
            c2.text_area("OCR text", text, height=200)
            st.success(f"Extracted: {', '.join(found) or 'nothing'}. Review the form below and fill the rest.")
        except Exception as e:
            st.error(f"OCR failed (is Tesseract installed?): {e}")

# ---- Path 2: manual input -> 13 clinical features ----
st.subheader("Patient data")
a, b, c = st.columns(3)
with a:
    age = st.number_input("Age", 20, 100, key="age")
    sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x else "Female", key="sex")
    cp = st.selectbox("Chest pain type", list(CP), format_func=CP.get, key="cp")
    trestbps = st.number_input("Resting BP (mm Hg)", 80, 220, key="trestbps")
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600, key="chol")
with b:
    fbs = st.selectbox("Fasting sugar > 120 mg/dl", [0, 1], format_func=lambda x: "Yes" if x else "No", key="fbs")
    restecg = st.selectbox("Resting ECG", list(ECG), format_func=ECG.get, key="restecg")
    thalach = st.number_input("Max heart rate", 60, 220, key="thalach")
    exang = st.selectbox("Exercise-induced angina", [0, 1], format_func=lambda x: "Yes" if x else "No", key="exang")
with c:
    oldpeak = st.number_input("ST depression", 0.0, 7.0, step=0.1, key="oldpeak")
    slope = st.selectbox("ST slope", list(SLOPE), format_func=SLOPE.get, key="slope")
    ca = st.selectbox("Major vessels (0–3)", [0, 1, 2, 3], key="ca")
    thal = st.selectbox("Thalassemia", list(THAL), format_func=THAL.get, key="thal")

if st.button("Predict risk", type="primary"):
    row = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                         thalach, exang, oldpeak, slope, ca, thal]], columns=COLS)
    p = float(model.predict_proba(row)[0, 1])
    high = p >= 0.5
    left, right = st.columns(2)
    with left:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100, number={"suffix": "%"},
            title={"text": "Risk probability"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#d62728" if high else "#2ca02c"},
                   "steps": [{"range": [0, 50], "color": "#e8f5e9"},
                             {"range": [50, 100], "color": "#fdecea"}]}))
        st.plotly_chart(fig, use_container_width=True)
        (st.error if high else st.success)(
            f"**{'Higher' if high else 'Lower'} risk** category. "
            "Please consult a doctor for proper evaluation.")
    with right:
        imp = pd.Series(model.named_steps["rf"].feature_importances_, index=COLS).sort_values()
        st.plotly_chart(go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h"))
                        .update_layout(title="Feature importance (model-wide)"),
                        use_container_width=True)
    if os.path.exists("model/metrics.json"):
        m = json.load(open("model/metrics.json"))
        st.caption(f"Model test accuracy {m['accuracy']:.2f} · ROC-AUC {m['roc_auc']:.2f}")
