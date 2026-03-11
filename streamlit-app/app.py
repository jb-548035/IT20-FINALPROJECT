"""
app.py
------
Streamlit web application for the Shopping Channel Preference Predictor.

INPUT FEATURES (10 total across 4 sections):
  Demographics        : age, monthly_income
  Digital Behaviour   : daily_internet_hours, social_media_hours, tech_savvy_score
  Shopping Behaviour  : monthly_online_orders, monthly_store_visits
  Attitudinal Factors : need_touch_feel_score, time_pressure_level

TWO MODES:
  • DEMO MODE  – Works immediately with no .pkl files (runs right now in PyCharm).
  • LIVE MODE  – Auto-activates when final_model.pkl exists in ../models/.

To run:
  cd streamlit-app
  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import sys
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
APP_DIR    = Path(__file__).parent
MODELS_DIR = APP_DIR.parent / "models"
sys.path.insert(0, str(APP_DIR))

from db import init_db, save_prediction, get_all_predictions, get_summary_stats, delete_all_predictions

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shopping Preference Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_db()

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    [data-testid="stAppViewContainer"] { background: #f4f7fb; }
    [data-testid="stSidebar"]          { background: #1a3a5c; }
    [data-testid="stSidebar"] * { color: #ffffff !important; }

    /* ── Header ── */
    .app-header {
        background: linear-gradient(135deg, #1a3a5c 0%, #2E86AB 100%);
        border-radius: 14px; padding: 2rem 2.5rem 1.5rem;
        text-align: center; margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .app-header h1 { color: #ffffff; font-size: 2rem; font-weight: 800; margin: 0 0 0.3rem; }
    .app-header p  { color: rgba(255,255,255,0.78); font-size: 1rem; margin: 0; }

    /* ── Section cards ── */
    .section-card {
        background: #ffffff; border-radius: 12px;
        padding: 1.4rem 1.6rem 1rem; margin-bottom: 1.2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.07);
        border-top: 4px solid #2E86AB;
    }
    .section-title {
        font-size: 1rem; font-weight: 700; color: #1a3a5c;
        margin-bottom: 1rem; letter-spacing: 0.02em;
    }

    /* ── Result card ── */
    .result-card {
        border-radius: 14px; padding: 2rem;
        text-align: center; color: #fff;
        box-shadow: 0 6px 24px rgba(0,0,0,0.18);
        margin-bottom: 1rem;
    }
    .result-class    { font-size: 2.4rem; font-weight: 900; margin: 0.4rem 0; }
    .result-conf     { font-size: 1.05rem; opacity: 0.88; }
    .result-model    { font-size: 0.78rem; opacity: 0.6; margin-top: 0.5rem; }

    /* ── Banners ── */
    .demo-banner {
        background: #fff8e1; border: 1.5px solid #ffc107;
        border-left: 5px solid #ffc107; border-radius: 8px;
        padding: 0.75rem 1rem; color: #664d03;
        font-size: 0.9rem; margin-bottom: 1rem;
    }
    .live-banner {
        background: #e8f5e9; border: 1.5px solid #43a047;
        border-left: 5px solid #43a047; border-radius: 8px;
        padding: 0.75rem 1rem; color: #1b5e20;
        font-size: 0.9rem; margin-bottom: 1rem;
    }

    /* ── Submit button ── */
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(90deg, #1a3a5c, #2E86AB) !important;
        color: #fff !important; font-size: 1.1rem !important;
        font-weight: 700 !important; border-radius: 10px !important;
        height: 3.2rem !important; border: none !important;
        box-shadow: 0 4px 14px rgba(46,134,171,0.4) !important;
        transition: opacity 0.2s !important;
    }
    div[data-testid="stFormSubmitButton"] > button:hover { opacity: 0.88 !important; }

    /* ── Metric boxes ── */
    [data-testid="stMetric"] {
        background: #fff; border-radius: 10px;
        padding: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        border-left: 4px solid #2E86AB;
    }

    /* ── Slider label fix ── */
    label { font-weight: 600 !important; color: #1a3a5c !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model_artifacts():
    """
    Tries to load the trained .pkl files from ../models/.
    Returns (model, scaler, le_target, le_cat, feature_info, is_demo).
    is_demo=True  →  Demo Mode  (heuristic, no .pkl files needed)
    is_demo=False →  Live Mode  (real Logistic Regression model)
    """
    required = [
        "final_model.pkl", "scaler.pkl",
        "label_encoder_target.pkl", "label_encoders_cat.pkl",
        "feature_info.json"
    ]
    all_present = all((MODELS_DIR / f).exists() for f in required)

    # Always try to load feature_info (ships with the repo)
    fi_path = MODELS_DIR / "feature_info.json"
    feature_info = json.load(open(fi_path)) if fi_path.exists() else _default_feature_info()

    if all_present:
        model     = joblib.load(MODELS_DIR / "final_model.pkl")
        scaler    = joblib.load(MODELS_DIR / "scaler.pkl")
        le_target = joblib.load(MODELS_DIR / "label_encoder_target.pkl")
        le_cat    = joblib.load(MODELS_DIR / "label_encoders_cat.pkl")
        return model, scaler, le_target, le_cat, feature_info, False   # LIVE

    return None, None, None, None, feature_info, True                  # DEMO


def _default_feature_info():
    return {
        "all_features": [
            "age", "monthly_income", "daily_internet_hours", "smartphone_usage_years",
            "social_media_hours", "online_payment_trust_score", "tech_savvy_score",
            "monthly_online_orders", "monthly_store_visits", "avg_online_spend",
            "avg_store_spend", "discount_sensitivity", "return_frequency",
            "avg_delivery_days", "delivery_fee_sensitivity", "free_return_importance",
            "product_availability_online", "impulse_buying_score", "need_touch_feel_score",
            "brand_loyalty_score", "environmental_awareness", "time_pressure_level",
            "gender", "city_tier",
            "online_to_store_spend_ratio", "digital_engagement_score",
            "store_preference_indicator", "income_spend_efficiency", "price_sensitivity_index"
        ],
        "categorical_features": ["gender", "city_tier"],
        "target_classes": ["Hybrid", "Online", "Store"]
    }


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING — fills in the non-input features with sensible defaults
# so the full 29-feature vector the trained model expects is always complete.
# ══════════════════════════════════════════════════════════════════════════════

def build_full_feature_dict(inputs: dict) -> dict:
    """
    Takes the 10 user-provided inputs and fills in the remaining features
    with dataset-average defaults, then engineers the 5 derived features.
    This keeps the trained model's feature vector complete.
    """
    # ── Dataset-average defaults for non-input features ──────────────────────
    defaults = {
        "smartphone_usage_years"     : 7.0,
        "online_payment_trust_score" : 6.5,
        "avg_online_spend"           : 3200.0,
        "avg_store_spend"            : 800.0,
        "discount_sensitivity"       : 6.0,
        "return_frequency"           : 4.0,
        "avg_delivery_days"          : 4.0,
        "delivery_fee_sensitivity"   : 5.5,
        "free_return_importance"     : 6.5,
        "product_availability_online": 7.0,
        "impulse_buying_score"       : 5.5,
        "brand_loyalty_score"        : 5.5,
        "environmental_awareness"    : 6.0,
        "gender"                     : "Male",
        "city_tier"                  : "Tier 1",
    }

    data = {**defaults, **inputs}   # user inputs override defaults

    # ── Engineered features (must match data_preprocessing.ipynb) ─────────────
    data["online_to_store_spend_ratio"] = (
        data["avg_online_spend"] / (data["avg_store_spend"] + 1)
    )
    data["digital_engagement_score"] = (
        data["daily_internet_hours"]  * 0.3 +
        data["social_media_hours"]    * 0.2 +
        data["tech_savvy_score"]      * 0.3 +
        data["monthly_online_orders"] * 0.2
    )
    data["store_preference_indicator"] = (
        data["need_touch_feel_score"]  * 0.4 +
        data["monthly_store_visits"]   * 0.4 +
        data["brand_loyalty_score"]    * 0.2
    )
    data["income_spend_efficiency"] = (
        data["monthly_income"] /
        (data["avg_online_spend"] + data["avg_store_spend"] + 1)
    )
    data["price_sensitivity_index"] = (
        data["discount_sensitivity"]     * 0.5 +
        data["delivery_fee_sensitivity"] * 0.5
    )
    return data


# ══════════════════════════════════════════════════════════════════════════════
# DEMO MODE — heuristic classifier (no .pkl files needed)
# ══════════════════════════════════════════════════════════════════════════════

def demo_predict(inputs: dict) -> dict:
    """
    Rule-based classifier that uses only the 10 user inputs to approximate
    what the trained Logistic Regression would predict.
    """
    age                    = inputs["age"]
    monthly_income         = inputs["monthly_income"]
    daily_internet_hours   = inputs["daily_internet_hours"]
    social_media_hours     = inputs["social_media_hours"]
    tech_savvy_score       = inputs["tech_savvy_score"]
    monthly_online_orders  = inputs["monthly_online_orders"]
    monthly_store_visits   = inputs["monthly_store_visits"]
    need_touch_feel_score  = inputs["need_touch_feel_score"]
    time_pressure_level    = inputs["time_pressure_level"]

    # Digital engagement composite
    digital_score = (
        daily_internet_hours  * 0.35 +
        social_media_hours    * 0.20 +
        tech_savvy_score      * 0.30 +
        monthly_online_orders * 0.15
    )

    # ── Score each class ──────────────────────────────────────────────────────
    online_score = (
        monthly_online_orders          * 0.40 +
        digital_score                  * 0.30 +
        tech_savvy_score               * 0.20 +
        time_pressure_level            * 0.10 +
        (10 - need_touch_feel_score)   * 0.15 +
        (monthly_income / 10000)       * 0.05
    )

    store_score = (
        monthly_store_visits           * 0.55 +
        need_touch_feel_score          * 0.35 +
        (10 - tech_savvy_score)        * 0.15 +
        (10 - daily_internet_hours / 2)* 0.10 +
        (age / 10)                     * 0.05
    )

    # Hybrid: customer uses both channels meaningfully
    both_active   = min(monthly_online_orders, monthly_store_visits * 3)
    hybrid_score  = both_active * 0.30 + 4.5   # base prior

    # ── Softmax probabilities ─────────────────────────────────────────────────
    scores = np.clip(
        np.array([hybrid_score, online_score, store_score], dtype=float), 0, None
    )
    probs  = scores / (scores.sum() + 1e-9)

    classes   = ["Hybrid", "Online", "Store"]
    pred_idx  = int(np.argmax(probs))
    pred_cls  = classes[pred_idx]
    conf      = min(0.99, float(probs[pred_idx]) * 1.25)

    return {
        "predicted_class": pred_cls,
        "confidence"     : conf,
        "probabilities"  : dict(zip(classes, [round(float(p), 4) for p in probs])),
        "model_used"     : "Demo Heuristic (run Colab notebooks for the real model)"
    }


# ══════════════════════════════════════════════════════════════════════════════
# LIVE MODE — real scikit-learn pipeline
# ══════════════════════════════════════════════════════════════════════════════

def live_predict(inputs: dict, model, scaler, le_target, le_cat, feature_info) -> dict:
    """Builds the full 29-feature vector, scales it, and runs the trained model."""
    data = build_full_feature_dict(inputs)

    # Encode categoricals
    for col in feature_info["categorical_features"]:
        le  = le_cat[col]
        val = str(data[col])
        data[col] = int(le.transform([val])[0]) if val in le.classes_ else 0

    vector = [data[f] for f in feature_info["all_features"]]
    df_row = pd.DataFrame([vector], columns=feature_info["all_features"])
    scaled = scaler.transform(df_row)

    pred_enc = model.predict(scaled)[0]
    proba    = model.predict_proba(scaled)[0]
    pred_cls = le_target.inverse_transform([pred_enc])[0]
    classes  = le_target.classes_.tolist()

    return {
        "predicted_class": pred_cls,
        "confidence"     : float(np.max(proba)),
        "probabilities"  : dict(zip(classes, [round(float(p), 4) for p in proba])),
        "model_used"     : type(model).__name__
    }


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ARTIFACTS (once, cached by Streamlit)
# ══════════════════════════════════════════════════════════════════════════════

model, scaler, le_target, le_cat, feature_info, IS_DEMO = load_model_artifacts()

# Colour & emoji maps shared across pages
COLORS = {"Online": "#2E86AB", "Store": "#E8A838", "Hybrid": "#28A745"}
EMOJI  = {"Online": "💻",      "Store": "🏪",      "Hybrid": "🔄"}
GRADIENT = {
    "Online": "linear-gradient(135deg,#2E86AB,#1a3a5c)",
    "Store" : "linear-gradient(135deg,#E8A838,#7a4e00)",
    "Hybrid": "linear-gradient(135deg,#28A745,#0a3d1f)",
}


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("## 🛒 Shopping Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🎯 Predict", "📊 Prediction History", "ℹ️ About"],
    label_visibility="collapsed"
)
st.sidebar.markdown("---")

if IS_DEMO:
    st.sidebar.warning(
        "**⚠️ Demo Mode**\n\n"
        "Model files not found in `../models/`.\n\n"
        "Run the 3 Colab notebooks then download the `models/` folder to activate Live Mode."
    )
else:
    st.sidebar.success(
        "**✅ Live Mode**\n\n"
        "Real trained model loaded and ready."
    )

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model:** Logistic Regression  \n"
    "**Accuracy:** ~98.2%  \n"
    "**Dataset:** 11,789 customers  \n"
    "**Classes:** Online · Store · Hybrid"
)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════

if page == "🎯 Predict":

    # Header
    st.markdown("""
    <div class="app-header">
        <h1>🛒 Shopping Channel Preference Predictor</h1>
        <p>Enter a customer profile to classify their preferred shopping channel:
           <strong>Online</strong>, <strong>In-Store</strong>, or <strong>Hybrid</strong></p>
    </div>
    """, unsafe_allow_html=True)

    # Mode banner
    if IS_DEMO:
        st.markdown("""
        <div class="demo-banner">
            ⚠️ <strong>Demo Mode Active</strong> —
            <code>final_model.pkl</code> not found in <code>../models/</code>.
            The app uses a built-in heuristic so you can demo the full UI right now.
            Run the 3 Colab notebooks and copy the <code>models/</code> folder here to switch to Live Mode.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="live-banner">
            ✅ <strong>Live Mode Active</strong> —
            Real trained Logistic Regression model loaded (~98.2% accuracy on held-out test set).
        </div>""", unsafe_allow_html=True)

    # ── Input Form ─────────────────────────────────────────────────────────────
    with st.form("prediction_form", clear_on_submit=False):

        # ── Section 1: Demographics ───────────────────────────────────────────
        st.markdown("""
        <div class="section-card">
            <div class="section-title">👤 Section 1 — Demographic Information</div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input(
                "Age (years)",
                min_value=10, max_value=100, value=28, step=1,
                help="Customer's age in years"
            )
        with c2:
            monthly_income = st.number_input(
                "Monthly Income (₹ INR)",
                min_value=0, max_value=1_000_000, value=45000, step=1000,
                help="Customer's gross monthly income in Indian Rupees"
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Section 2: Digital Behaviour ──────────────────────────────────────
        st.markdown("""
        <div class="section-card">
            <div class="section-title">📱 Section 2 — Digital Behaviour</div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            daily_internet_hours = st.slider(
                "Daily Internet Hours",
                min_value=0.0, max_value=24.0, value=5.5, step=0.5,
                help="Average hours spent online per day (0 = never, 24 = always connected)"
            )
        with c2:
            social_media_hours = st.slider(
                "Social Media Hours / Day",
                min_value=0.0, max_value=12.0, value=2.5, step=0.5,
                help="Average daily hours on social media platforms"
            )
        with c3:
            tech_savvy_score = st.slider(
                "Tech Savvy Score  (1 – 10)",
                min_value=1, max_value=10, value=7,
                help="1 = struggles with technology  ·  10 = very comfortable with technology"
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Section 3: Shopping Behaviour ────────────────────────────────────
        st.markdown("""
        <div class="section-card">
            <div class="section-title">🛍️ Section 3 — Shopping Behaviour</div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            monthly_online_orders = st.number_input(
                "Monthly Online Orders",
                min_value=0, max_value=100, value=12, step=1,
                help="Number of online orders placed per month"
            )
        with c2:
            monthly_store_visits = st.number_input(
                "Monthly Store Visits",
                min_value=0, max_value=50, value=2, step=1,
                help="Number of times the customer visits a physical store per month"
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Section 4: Attitudinal Factors ────────────────────────────────────
        st.markdown("""
        <div class="section-card">
            <div class="section-title">🧠 Section 4 — Attitudinal Factors</div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            need_touch_feel_score = st.slider(
                "Need Touch / Feel Score  (1 – 10)",
                min_value=1, max_value=10, value=3,
                help="1 = fine buying without seeing  ·  10 = must see/touch before buying"
            )
        with c2:
            time_pressure_level = st.slider(
                "Time Pressure Level  (1 – 10)",
                min_value=1, max_value=10, value=8,
                help="1 = lots of free time  ·  10 = extremely time-pressured (prefers fast delivery)"
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Submit ────────────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "🔮  Predict Shopping Preference",
            use_container_width=True,
            type="primary"
        )

    # ── Handle Submission ──────────────────────────────────────────────────────
    if submitted:
        inputs = {
            "age"                  : age,
            "monthly_income"       : monthly_income,
            "daily_internet_hours" : daily_internet_hours,
            "social_media_hours"   : social_media_hours,
            "tech_savvy_score"     : tech_savvy_score,
            "monthly_online_orders": monthly_online_orders,
            "monthly_store_visits" : monthly_store_visits,
            "need_touch_feel_score": need_touch_feel_score,
            "time_pressure_level"  : time_pressure_level,
        }

        with st.spinner("Running prediction..."):
            if IS_DEMO:
                result = demo_predict(inputs)
            else:
                result = live_predict(inputs, model, scaler, le_target, le_cat, feature_info)

        # Save to SQLite
        save_prediction(
            result["predicted_class"], result["confidence"],
            result["probabilities"],   result["model_used"], inputs
        )

        pred  = result["predicted_class"]
        conf  = result["confidence"]
        proba = result["probabilities"]

        st.markdown("---")
        st.markdown("## 🎯 Prediction Result")

        col_card, col_chart = st.columns([1, 1])

        # ── Result card ────────────────────────────────────────────────────────
        with col_card:
            insights = {
                "Online":
                    "💻 This customer strongly prefers **online shopping**.  \n\n"
                    "**Recommended actions:**  \n"
                    "- Invest in personalised digital recommendations  \n"
                    "- Offer fast-delivery options and app push notifications  \n"
                    "- Streamline mobile checkout experience",
                "Store":
                    "🏪 This customer prefers the **in-store experience**.  \n\n"
                    "**Recommended actions:**  \n"
                    "- Focus on in-store promotions and loyalty cards  \n"
                    "- Provide personalised in-store assistance  \n"
                    "- Enhance visual merchandising and store layout",
                "Hybrid":
                    "🔄 This customer actively uses **both channels**.  \n\n"
                    "**Recommended actions:**  \n"
                    "- Deploy omni-channel strategy (click-and-collect)  \n"
                    "- Unify loyalty points across channels  \n"
                    "- Use cross-channel personalised recommendations",
            }

            st.markdown(f"""
            <div class="result-card" style="background:{GRADIENT[pred]};">
                <div style="font-size:0.85rem;opacity:0.75;letter-spacing:1.5px;text-transform:uppercase;">
                    Predicted Shopping Preference
                </div>
                <div class="result-class">{EMOJI[pred]}&nbsp;{pred}</div>
                <div class="result-conf">Confidence: <strong>{conf*100:.1f}%</strong></div>
                <div class="result-model">Model: {result["model_used"]}</div>
            </div>
            """, unsafe_allow_html=True)

            st.info(insights[pred])

        # ── Probability chart ──────────────────────────────────────────────────
        with col_chart:
            st.markdown("**📊 Class Probability Distribution**")

            classes = list(proba.keys())
            values  = [proba[c] * 100 for c in classes]
            bar_colors = [COLORS.get(c, "#888") for c in classes]

            fig, ax = plt.subplots(figsize=(6, 3.5))
            bars = ax.barh(classes, values, color=bar_colors, edgecolor="white", height=0.55)
            ax.set_xlim(0, 118)
            ax.set_xlabel("Probability (%)", fontsize=10)
            ax.set_title("Prediction Probabilities", fontweight="bold", fontsize=11)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="y", labelsize=11)

            for bar, v in zip(bars, values):
                ax.text(
                    v + 1.5, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}%", va="center", fontweight="bold", fontsize=10
                )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Probability breakdown
            st.markdown("**Detailed breakdown:**")
            for cls, prob in sorted(proba.items(), key=lambda x: -x[1]):
                marker = " ✅" if cls == pred else ""
                st.markdown(f"`{cls}{marker}` &nbsp; **{prob*100:.1f}%**")
                st.progress(float(prob))

        # ── Input summary expander ────────────────────────────────────────────
        with st.expander("📋 View submitted customer profile"):
            summary = {
                "Age"                  : f"{age} yrs",
                "Monthly Income"       : f"₹{monthly_income:,}",
                "Daily Internet Hours" : f"{daily_internet_hours} hrs",
                "Social Media Hours"   : f"{social_media_hours} hrs/day",
                "Tech Savvy Score"     : f"{tech_savvy_score}/10",
                "Monthly Online Orders": monthly_online_orders,
                "Monthly Store Visits" : monthly_store_visits,
                "Need Touch/Feel Score": f"{need_touch_feel_score}/10",
                "Time Pressure Level"  : f"{time_pressure_level}/10",
            }
            st.table(pd.DataFrame(summary.items(), columns=["Feature", "Value"]))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICTION HISTORY
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Prediction History":
    st.title("📊 Prediction History")
    st.markdown("All predictions saved in this and previous sessions.")

    stats = get_summary_stats()
    dist  = stats.get("class_distribution", {})

    # Summary metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("🔢 Total Predictions", stats["total_predictions"])
    with c2:
        avg_c = (stats["avg_confidence"] or 0) * 100
        st.metric("🎯 Average Confidence", f"{avg_c:.1f}%")
    with c3:
        most = max(dist, key=dist.get) if dist else "N/A"
        st.metric("🏆 Most Predicted Class", most)

    # Class distribution chart
    if dist:
        st.markdown("### Class Distribution")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(
            dist.keys(), dist.values(),
            color=[COLORS.get(k, "#888") for k in dist],
            edgecolor="white", width=0.45
        )
        ax.set_ylabel("Count")
        ax.set_title("Prediction Class Distribution", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        for i, (k, v) in enumerate(dist.items()):
            ax.text(i, v + 0.1, str(v), ha="center", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Table
    st.markdown("### Recent Predictions (latest 50)")
    rows = get_all_predictions(limit=50)
    if rows:
        st.dataframe(
            pd.DataFrame([{
                "#"          : r["id"],
                "Timestamp"  : r["timestamp"],
                "Prediction" : r["predicted_class"],
                "Confidence" : f"{r['confidence'] * 100:.1f}%",
                "Model"      : r["model_used"],
            } for r in rows]),
            use_container_width=True,
            hide_index=True
        )
        if st.button("🗑️ Clear All History", type="secondary"):
            delete_all_predictions()
            st.success("History cleared!")
            st.rerun()
    else:
        st.info("No predictions yet — head to the Predict page to make your first one!")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")

    col_main, col_side = st.columns([3, 2])

    with col_main:
        st.markdown("""
## Predicting Consumer Shopping Channel Preference
### A Machine Learning Classification Framework
---

### 🎯 Business Problem
An omni-channel retailer lacks a data-driven mechanism to predict customer shopping
preferences. Without classification, marketing budgets are allocated generically,
failing to optimise the customer journey for high-value segments.

### 1.1 Objectives
| # | Objective |
|---|-----------|
| 1.1.1 | Analyse the dataset and define a classification problem (Online / Store / Hybrid) |
| 1.1.2 | Implement supervised ML models to predict shopping preferences |
| 1.1.3 | Compare human-selected vs AI-recommended models on accuracy and usability |
| 1.1.4 | Conduct comparative performance analysis balancing precision with interpretability |

### 📥 Input Features Used (10)
| Section | Feature | Description |
|---------|---------|-------------|
| Demographics | `age` | Customer age |
| Demographics | `monthly_income` | Gross monthly income (₹) |
| Digital Behaviour | `daily_internet_hours` | Hours online per day |
| Digital Behaviour | `social_media_hours` | Social media usage per day |
| Digital Behaviour | `tech_savvy_score` | Technology comfort (1–10) |
| Shopping Behaviour | `monthly_online_orders` | Online orders per month |
| Shopping Behaviour | `monthly_store_visits` | Physical store visits per month |
| Attitudinal | `need_touch_feel_score` | Need to see/touch before buying (1–10) |
| Attitudinal | `time_pressure_level` | Time pressure (1 = relaxed, 10 = very busy) |

### 🤖 Model Comparison
| Model | Type | Accuracy | Weighted F1 |
|-------|------|----------|-------------|
| **Logistic Regression ✅** | Human-selected (Final) | **98.2%** | **0.980** |
| SVM (RBF) | Benchmark | 96.4% | 0.949 |
| XGBoost | AI-recommended | ~96.0% | ~0.955 |
| Gradient Boosting | Benchmark | 93.0% | 0.914 |
| Random Forest | Benchmark | 91.1% | 0.886 |

### 📊 Dataset
- **Source:** Kaggle – Online vs In-Store Shopping Behaviour
- **Records:** 11,789 customers · **Features:** 26 · **Target:** shopping_preference
        """)

    with col_side:
        st.markdown("""
### 🔑 Top Predictive Features
1. `monthly_online_orders`
2. `avg_online_spend`
3. `online_to_store_spend_ratio` *(engineered)*
4. `tech_savvy_score`
5. `need_touch_feel_score`
6. `daily_internet_hours`
7. `digital_engagement_score` *(engineered)*

---
### ⚙️ Current Mode
        """)

        if IS_DEMO:
            st.warning(
                "**⚠️ Demo Mode**\n\n"
                "Running built-in heuristic.\n\n"
                "Run the 3 Colab notebooks and copy `models/` here to switch to Live Mode."
            )
        else:
            st.success(
                "**✅ Live Mode**\n\n"
                "Real trained Logistic Regression model is active."
            )

        st.markdown("""
---
### 📁 Project Layout
```
shopping-project/
├── ml-training/
│   └── notebooks/
│       ├── data_preprocessing.ipynb
│       ├── model_selection.ipynb
│       ├── hyperparameter_tuning.ipynb
│       └── model_testing.ipynb
├── models/        ← .pkl files go here
├── py-app/        ← FastAPI REST API
├── streamlit-app/ ← You are here
└── web-app/       ← ASP.NET Core MVC
```

---
### 👥 Team
Machine Learning Classification Project  
Built with Python · scikit-learn · Streamlit
        """)