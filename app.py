import inspect
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION (must be the first Streamlit call)
# ============================================================

st.set_page_config(
    page_title="University Admission Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# HELPERS
# ============================================================

def render_html(markup: str) -> None:
    """Render an HTML snippet safely through st.markdown.

    Markdown treats lines indented by 4+ spaces (and anything after a
    blank line) as a code block, which is why HTML was showing up as
    raw text. Stripping indentation and blank lines avoids that.
    """
    cleaned = "\n".join(
        line.strip() for line in markup.splitlines() if line.strip()
    )
    st.markdown(cleaned, unsafe_allow_html=True)


def _stretch(func) -> dict:
    """Full-width kwarg that works on both old and new Streamlit versions."""
    if "width" in inspect.signature(func).parameters:
        return {"width": "stretch"}
    return {"use_container_width": True}


# ============================================================
# CUSTOM CSS
# ============================================================

render_html(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #eef5ff 0%, #f8fbff 50%, #eaf2ff 100%);
    }
    .block-container {
        max-width: 1200px;
        padding-top: 4rem;
        padding-bottom: 3rem;
    }
    .main-header {
        background: linear-gradient(135deg, #0b3a82, #1769aa);
        padding: 35px;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0px 8px 25px rgba(11, 58, 130, 0.25);
        margin-bottom: 25px;
    }
    .main-header .header-icon {
        font-size: 48px;
        line-height: 1;
        margin-bottom: 10px;
    }
    .main-header h1 {
        font-size: 42px;
        margin: 0 0 10px 0;
        padding: 0;
        color: white;
    }
    .main-header p {
        font-size: 18px;
        margin: 0;
        color: white;
    }
    .card {
        background: white;
        color: #1f2937;
        padding: 25px;
        border-radius: 18px;
        box-shadow: 0px 5px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
    }
    .card h2 {
        color: #0b3a82;
        margin-top: 0;
    }
    .result-card {
        background: linear-gradient(135deg, #0b3a82, #145da0);
        color: white;
        padding: 35px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0px 8px 25px rgba(11, 58, 130, 0.25);
        margin-top: 25px;
        margin-bottom: 20px;
    }
    .result-card h2 {
        font-size: 25px;
        margin-bottom: 10px;
        color: white;
    }
    .result-card p {
        color: white;
    }
    .percentage {
        font-size: 58px;
        font-weight: bold;
        margin: 10px 0;
    }
    .feature-box {
        background: white;
        color: #1f2937;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #1769aa;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.06);
        margin-bottom: 12px;
        min-height: 120px;
    }
    .feature-box h4 {
        color: #0b3a82;
        margin: 0 0 8px 0;
    }
    .stButton > button {
        border-radius: 12px;
        height: 50px;
        font-size: 18px;
        font-weight: bold;
    }
    [data-testid="stMetric"] {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.07);
    }
    [data-testid="stMetric"] * {
        color: #1f2937;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b3a82, #062858);
    }
    section[data-testid="stSidebar"] * {
        color: white;
    }
    </style>
    """
)


# ============================================================
# MODEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Admition_prediction.pkl")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Unable to load model: {e}")
        return None


model = load_model()

if model is None:
    st.error(
        "❌ Model file not found or could not be loaded.\n\n"
        "Please make sure `Admition_prediction.pkl` is in the same "
        "folder as `app.py`."
    )
    st.stop()


def build_model_input(model, values: dict) -> pd.DataFrame:
    """Build the input frame and match the model's training columns.

    The public Graduate Admissions dataset has columns such as 'LOR '
    (with a trailing space), so we match names ignoring case and
    surrounding whitespace, then reorder to the training order.
    """
    df = pd.DataFrame({k: [v] for k, v in values.items()})

    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        return df

    lookup = {c.strip().lower(): c for c in df.columns}
    ordered = {}
    for col in expected:
        key = str(col).strip().lower()
        if key not in lookup:
            raise KeyError(f"Model expects column '{col}' which is not in the input.")
        ordered[col] = df[lookup[key]]
    return pd.DataFrame(ordered)


# ============================================================
# FORM STATE (keys let the Reset button work reliably)
# ============================================================

DEFAULTS = {
    "gre": 300,
    "toefl": 100,
    "cgpa": 8.0,
    "university_rating": 3,
    "sop": 3.0,
    "lor": 3.0,
    "research": 0,
}

for _key, _val in DEFAULTS.items():
    st.session_state.setdefault(_key, _val)


def reset_form() -> None:
    for key, val in DEFAULTS.items():
        st.session_state[key] = val
    st.session_state.pop("result", None)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🎓 Admission AI")
    st.markdown("---")
    st.markdown(
        """
### 🤖 About the Application

This application uses a machine learning regression model to
estimate a student's admission probability.

### 📌 Input Features

- GRE Score
- TOEFL Score
- University Rating
- SOP
- LOR
- CGPA
- Research Experience
"""
    )
    st.markdown("---")
    st.markdown(
        """
### 📊 Target

**Chance of Admit**

The model predicts a value between **0 and 1**.

Example: **0.80 = approximately 80%**
"""
    )
    st.markdown("---")
    st.caption("Machine Learning University Admission Prediction System")


# ============================================================
# HEADER
# ============================================================

render_html(
    """
    <div class="main-header">
        <div class="header-icon">🎓</div>
        <div class="header-content">
            <h1>University Admission Predictor</h1>
            <p>Predict the estimated admission probability using academic and profile information.</p>
        </div>
    </div>
    """
)


# ============================================================
# INTRODUCTION
# ============================================================

render_html(
    """
    <div class="card">
        <h2>📊 Student Profile</h2>
        <p>
            Enter the student's academic and profile information below.
            The machine learning model will estimate the student's
            admission probability.
        </p>
    </div>
    """
)


# ============================================================
# INPUT SECTION
# ============================================================

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📚 Academic Scores")

    gre = st.number_input(
        "GRE Score",
        min_value=260,
        max_value=340,
        step=1,
        key="gre",
        help="Graduate Record Examination score. Typical range: 260–340.",
    )

    toefl = st.number_input(
        "TOEFL Score",
        min_value=0,
        max_value=120,
        step=1,
        key="toefl",
        help="English language proficiency score. Typical range: 0–120.",
    )

    cgpa = st.number_input(
        "CGPA",
        min_value=0.0,
        max_value=10.0,
        step=0.01,
        format="%.2f",
        key="cgpa",
        help="Cumulative Grade Point Average on a 10-point scale.",
    )

with col2:
    st.markdown("### 🏫 University & Profile")

    university_rating = st.slider(
        "University Rating",
        min_value=1,
        max_value=5,
        step=1,
        key="university_rating",
        help="University rating from 1 to 5.",
    )

    sop = st.slider(
        "SOP Rating",
        min_value=1.0,
        max_value=5.0,
        step=0.5,
        key="sop",
        help="Statement of Purpose rating from 1 to 5.",
    )

    lor = st.slider(
        "LOR Rating",
        min_value=1.0,
        max_value=5.0,
        step=0.5,
        key="lor",
        help="Letter of Recommendation rating from 1 to 5.",
    )

    research = st.selectbox(
        "Research Experience",
        options=[0, 1],
        format_func=lambda x: "✅ Yes" if x == 1 else "❌ No",
        key="research",
        help="0 = No research, 1 = Research experience",
    )


# ============================================================
# INPUT SUMMARY
# ============================================================

st.markdown("---")
st.subheader("📋 Student Profile Summary")

s1, s2, s3, s4 = st.columns(4)
s1.metric("GRE Score", gre)
s2.metric("TOEFL Score", toefl)
s3.metric("CGPA", f"{cgpa:.2f}")
s4.metric("Research", "Yes" if research == 1 else "No")


# ============================================================
# BUTTONS
# ============================================================

st.markdown("---")

b1, b2 = st.columns(2)

with b1:
    predict_button = st.button(
        "🔮 Predict Admission Chance",
        type="primary",
        **_stretch(st.button),
    )

with b2:
    st.button(
        "🔄 Reset",
        on_click=reset_form,
        **_stretch(st.button),
    )


# ============================================================
# PREDICTION (result is stored so it survives widget changes)
# ============================================================

if predict_button:
    try:
        input_data = build_model_input(
            model,
            {
                "GRE Score": gre,
                "TOEFL Score": toefl,
                "University Rating": university_rating,
                "SOP": sop,
                "LOR": lor,
                "CGPA": cgpa,
                "Research": research,
            },
        )

        raw_prediction = float(np.ravel(model.predict(input_data))[0])
        st.session_state["result"] = {
            "prediction": float(np.clip(raw_prediction, 0.0, 1.0)),
            "input_data": input_data,
        }

    except Exception as e:
        st.session_state.pop("result", None)
        st.error("❌ Prediction failed.")
        st.exception(e)
        st.info(
            "**Important:** The input columns used by the application must "
            "match the columns used when training `Admition_prediction.pkl`.\n\n"
            "Expected columns: GRE Score, TOEFL Score, University Rating, "
            "SOP, LOR, CGPA, Research"
        )


# ============================================================
# RESULT
# ============================================================

result = st.session_state.get("result")

if result:
    prediction = result["prediction"]
    percentage = prediction * 100

    render_html(
        f"""
        <div class="result-card">
            <h2>🎯 Predicted Admission Chance</h2>
            <div class="percentage">{percentage:.2f}%</div>
            <p>Estimated probability based on the provided academic profile.</p>
        </div>
        """
    )

    st.progress(prediction, text=f"Admission Probability: {percentage:.2f}%")

    if prediction >= 0.80:
        st.success("🟢 High predicted admission probability.")
    elif prediction >= 0.60:
        st.info("🔵 Moderate predicted admission probability.")
    else:
        st.warning("🟠 Lower predicted admission probability.")

    st.markdown("---")
    st.subheader("📊 Prediction Details")

    r1, r2, r3 = st.columns(3)
    r1.metric("Probability", f"{prediction:.4f}")
    r2.metric("Percentage", f"{percentage:.2f}%")
    r3.metric("Research", "Yes" if research == 1 else "No")

    with st.expander("🔍 View Model Input"):
        st.dataframe(
            result["input_data"],
            hide_index=True,
            **_stretch(st.dataframe),
        )


# ============================================================
# FEATURE INFORMATION
# ============================================================

FEATURES_LEFT = [
    ("GRE Score",
     "Graduate Record Examination (GRE) score of the student, typically "
     "ranging from 260–340.<br><br>A higher score generally indicates "
     "stronger GRE performance."),
    ("TOEFL Score",
     "Test of English as a Foreign Language (TOEFL) score.<br><br>"
     "Typically ranges from 0–120 and measures English-language proficiency."),
    ("University Rating",
     "Rating of the university the student is applying to.<br><br>"
     "Commonly represented on a 1–5 scale."),
    ("SOP",
     "Statement of Purpose rating.<br><br>Usually represented on a 1–5 "
     "scale and reflects the quality or strength of the student's SOP."),
]

FEATURES_RIGHT = [
    ("LOR",
     "Letter of Recommendation rating.<br><br>Usually represented on a "
     "1–5 scale and reflects the strength of recommendations provided by "
     "professors or professionals."),
    ("CGPA",
     "Cumulative Grade Point Average of the student.<br><br>Commonly "
     "measured on a 10-point scale. A higher CGPA indicates stronger "
     "academic performance."),
    ("Research",
     "Indicates whether the student has research experience.<br><br>"
     "<strong>0 = No</strong><br><strong>1 = Yes</strong>"),
    ("Chance of Admit",
     "The target variable predicted by the regression model.<br><br>"
     "Example: <strong>0.80 = approximately 80%</strong>"),
]


def feature_box(title: str, body: str) -> None:
    render_html(
        f"""
        <div class="feature-box">
            <h4>{title}</h4>
            {body}
        </div>
        """
    )


st.markdown("---")

with st.expander("📖 Understand the Input Features"):
    fc1, fc2 = st.columns(2)
    with fc1:
        for title, body in FEATURES_LEFT:
            feature_box(title, body)
    with fc2:
        for title, body in FEATURES_RIGHT:
            feature_box(title, body)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

render_html(
    """
    <div style="text-align:center; color:#555; padding:20px;">
        <strong>🎓 University Admission Predictor</strong>
        <br><br>
        Powered by Machine Learning
        <br><br>
        <small>
            The prediction is an estimate generated by the trained machine
            learning model and is not an actual university admission decision.
        </small>
    </div>
    """
)