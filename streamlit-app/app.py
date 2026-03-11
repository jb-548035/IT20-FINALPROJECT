"""
app.py
------
Streamlit web application for the Shopping Channel Preference Predictor.

Runs the ML model DIRECTLY (no separate API needed) for simplicity.
The model artifacts (final_model.pkl, scaler.pkl, etc.) must exist in ../models/.

To run:
  cd streamlit-app
  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

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

# ─── Initialize database on first run ─────────────────────────────────────────
init_db()


# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 800;
        color: #1a1a2e; text-align: center; margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem; color: #555; text-align: center; margin-bottom: 2rem;
    }
    .pred-card {
        background: linear-gradient(135deg, #4C72B0, #1a1a2e);
        color: white; padding: 1.5rem; border-radius: 12px;
        text-align: center; margin: 1rem 0;
    }
    .pred-class { font-size: 2rem; font-weight: 800; margin: 0.5rem 0; }
    .pred-conf  { font-size: 1rem; opacity: 0.85; }
    .metric-box {
        background: #f8f9fa; border-radius: 10px; padding: 1rem;
        text-align: center; border-left: 4px solid #4C72B0;
    }
    .section-header {
        font-size: 1.1rem; font-weight: 700;
        color: #1a1a2e; margin-top: 1.5rem; margin-bottom: 0.5rem;
        border-bottom: 2px solid #4C72B0; padding-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Model Loading (cached) ────────────────────────────────────────────────────

@st.cache_resource
def load_model_artifacts():
    """
    Loads all model artifacts once and caches them in memory.
    Streamlit's @st.cache_resource keeps this alive across reruns.
    """
    model     = joblib.load(MODELS_DIR / "final_model.pkl")
    scaler    = joblib.load(MODELS_DIR / "scaler.pkl")
    le_target = joblib.load(MODELS_DIR / "label_encoder_target.pkl")
    le_cat    = joblib.load(MODELS_DIR / "label_encoders_cat.pkl")
    with open(MODELS_DIR / "feature_info.json") as f:
        feature_info = json.load(f)
    return model, scaler, le_target, le_cat, feature_info


def engineer_features(data: dict) -> dict:
    """Recreate the 5 engineered features (must match training)."""
    data["online_to_store_spend_ratio"] = data["avg_online_spend"] / (data["avg_store_spend"] + 1)
    data["digital_engagement_score"]    = (
        data["daily_internet_hours"] * 0.3 + data["social_media_hours"] * 0.2 +
        data["tech_savvy_score"] * 0.3 + data["monthly_online_orders"] * 0.2
    )
    data["store_preference_indicator"]  = (
        data["need_touch_feel_score"] * 0.4 + data["monthly_store_visits"] * 0.4 +
        data["brand_loyalty_score"] * 0.2
    )
    data["income_spend_efficiency"]     = data["monthly_income"] / (
        data["avg_online_spend"] + data["avg_store_spend"] + 1
    )
    data["price_sensitivity_index"]     = (
        data["discount_sensitivity"] * 0.5 + data["delivery_fee_sensitivity"] * 0.5
    )
    return data


def run_prediction(inputs: dict, model, scaler, le_target, le_cat, feature_info) -> dict:
    """Full preprocessing + prediction pipeline."""
    data = dict(inputs)
    data = engineer_features(data)

    # Encode categoricals
    for col in feature_info["categorical_features"]:
        le = le_cat[col]
        val = str(data[col])
        data[col] = int(le.transform([val])[0]) if val in le.classes_ else 0

    # Assemble feature vector in correct order
    feature_vector = [data[f] for f in feature_info["all_features"]]
    df_row = pd.DataFrame([feature_vector], columns=feature_info["all_features"])
    scaled = scaler.transform(df_row)

    # Predict
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


# ─── Sidebar Navigation ────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=80)
st.sidebar.title("🛒 Shopping Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🎯 Predict", "📊 Prediction History", "ℹ️ About"]
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model:** Logistic Regression\n\n"
    "**Dataset:** Online vs Store Shopping\n\n"
    "**Accuracy:** ~98% (test set)"
)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: PREDICT
# ══════════════════════════════════════════════════════════════════════════════

if page == "🎯 Predict":
    st.markdown('<div class="main-title">🛒 Shopping Channel Preference Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Predict whether a customer prefers Online, In-Store, or Hybrid shopping</div>', unsafe_allow_html=True)

    try:
        model, scaler, le_target, le_cat, feature_info = load_model_artifacts()
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please run the training notebooks (data_preprocessing.ipynb → model_selection.ipynb → hyperparameter_tuning.ipynb) first.")
        st.stop()

    st.markdown("---")

    # ── Input Form ─────────────────────────────────────────────────────────────
    with st.form("prediction_form"):

        st.markdown('<div class="section-header">👤 Demographic Information</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            age = st.number_input("age", min_value=10, max_value=100, value=28, step=1)
        with col2:
            monthly_income = st.number_input("Monthly Income (INR)", min_value=0, value=45000, step=1000)
        with col3:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        with col4:
            city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])

        st.markdown('<div class="section-header">📱 Digital Behavior</div>', unsafe_allow_html=True)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            daily_internet_hours = st.slider("Internet Hours/Day", 0.0, 24.0, 5.5, 0.5)
        with col2:
            smartphone_usage_years = st.number_input("Smartphone Usage (yrs)", 0.0, 20.0, 7.0, 0.5)
        with col3:
            social_media_hours = st.slider("Social Media Hours/Day", 0.0, 12.0, 2.5, 0.5)
        with col4:
            online_payment_trust_score = st.slider("Payment Trust Score", 1, 10, 8)
        with col5:
            tech_savvy_score = st.slider("Tech Savvy Score", 1, 10, 7)

        st.markdown('<div class="section-header">🛍️ Shopping Behavior</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            monthly_online_orders = st.number_input("Online Orders/Month", 0, 100, 12, 1)
        with col2:
            monthly_store_visits = st.number_input("Store Visits/Month", 0, 50, 2, 1)
        with col3:
            avg_online_spend = st.number_input("Avg Online Spend (INR)", 0, 50000, 3200, 100)
        with col4:
            avg_store_spend = st.number_input("Avg Store Spend (INR)", 0, 50000, 800, 100)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            discount_sensitivity = st.slider("Discount Sensitivity", 1, 10, 7)
        with col2:
            return_frequency = st.slider("Return Frequency", 1, 10, 4)
        with col3:
            avg_delivery_days = st.number_input("Avg Delivery Days", 0.0, 30.0, 3.0, 0.5)
        with col4:
            delivery_fee_sensitivity = st.slider("Delivery Fee Sensitivity", 1, 10, 6)

        st.markdown('<div class="section-header">🧠 Attitudinal Factors</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            free_return_importance = st.slider("Free Return Importance", 1, 10, 8)
        with col2:
            product_availability_online = st.slider("Online Availability Score", 1, 10, 8)
        with col3:
            impulse_buying_score = st.slider("Impulse Buying Score", 1, 10, 6)
        with col4:
            need_touch_feel_score = st.slider("Need Touch/Feel Score", 1, 10, 3)

        col1, col2, col3 = st.columns(3)
        with col1:
            brand_loyalty_score = st.slider("Brand Loyalty Score", 1, 10, 5)
        with col2:
            environmental_awareness = st.slider("Environmental Awareness", 1, 10, 7)
        with col3:
            time_pressure_level = st.slider("Time Pressure Level", 1, 10, 8)

        st.markdown("---")
        submitted = st.form_submit_button("🔮 Predict Shopping Preference", use_container_width=True)

    # ── Handle Submission ──────────────────────────────────────────────────────
    if submitted:
        inputs = {
            "age": age, "monthly_income": monthly_income,
            "gender": gender, "city_tier": city_tier,
            "daily_internet_hours": daily_internet_hours,
            "smartphone_usage_years": smartphone_usage_years,
            "social_media_hours": social_media_hours,
            "online_payment_trust_score": online_payment_trust_score,
            "tech_savvy_score": tech_savvy_score,
            "monthly_online_orders": monthly_online_orders,
            "monthly_store_visits": monthly_store_visits,
            "avg_online_spend": avg_online_spend,
            "avg_store_spend": avg_store_spend,
            "discount_sensitivity": discount_sensitivity,
            "return_frequency": return_frequency,
            "avg_delivery_days": avg_delivery_days,
            "delivery_fee_sensitivity": delivery_fee_sensitivity,
            "free_return_importance": free_return_importance,
            "product_availability_online": product_availability_online,
            "impulse_buying_score": impulse_buying_score,
            "need_touch_feel_score": need_touch_feel_score,
            "brand_loyalty_score": brand_loyalty_score,
            "environmental_awareness": environmental_awareness,
            "time_pressure_level": time_pressure_level,
        }

        with st.spinner("Running prediction..."):
            result = run_prediction(inputs, model, scaler, le_target, le_cat, feature_info)

        # ── Save to database ───────────────────────────────────────────────────
        save_prediction(
            result["predicted_class"], result["confidence"],
            result["probabilities"],   result["model_used"], inputs
        )

        # ── Display Results ────────────────────────────────────────────────────
        st.markdown("## 🎯 Prediction Result")
        col_res, col_prob = st.columns([1, 1])

        with col_res:
            emoji_map = {"Online": "💻", "Store": "🏪", "Hybrid": "🔄"}
            color_map = {"Online": "#4C72B0", "Store": "#DD8452", "Hybrid": "#55A868"}
            pred = result["predicted_class"]
            conf = result["confidence"]

            st.markdown(f"""
            <div class="pred-card" style="background: linear-gradient(135deg, {color_map[pred]}, #1a1a2e);">
                <div style="font-size:1rem; opacity:0.8;">Predicted Preference</div>
                <div class="pred-class">{emoji_map[pred]} {pred}</div>
                <div class="pred-conf">Confidence: {conf*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            # Insight text
            insights = {
                "Online": "This customer strongly prefers **online shopping**. "
                          "Invest in personalized digital recommendations, fast delivery, "
                          "and seamless mobile checkout to retain this customer.",
                "Store" : "This customer prefers **in-store shopping**. "
                          "Focus on enhancing the physical retail experience, "
                          "in-store promotions, and personalized service.",
                "Hybrid": "This customer is a **hybrid shopper**. "
                          "Use an omni-channel strategy — connect online browsing "
                          "with in-store fulfillment (e.g., click-and-collect)."
            }
            st.info(insights[pred])

        with col_prob:
            st.markdown("**Class Probability Distribution**")
            proba = result["probabilities"]

            fig, ax = plt.subplots(figsize=(6, 3.5))
            colors_map = {"Online": "#4C72B0", "Store": "#DD8452", "Hybrid": "#55A868"}
            classes = list(proba.keys())
            values  = [proba[c] * 100 for c in classes]
            bars = ax.barh(classes, values,
                           color=[colors_map.get(c, "#888") for c in classes],
                           edgecolor='white', height=0.5)
            ax.set_xlim(0, 110)
            ax.set_xlabel("Probability (%)")
            ax.set_title("Prediction Probabilities")
            for bar, v in zip(bars, values):
                ax.text(v + 1, bar.get_y() + bar.get_height()/2,
                        f'{v:.1f}%', va='center', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: PREDICTION HISTORY
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Prediction History":
    st.title("📊 Prediction History")
    st.markdown("All predictions saved during this session and previous sessions.")

    stats = get_summary_stats()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Predictions", stats["total_predictions"])
    with col2:
        st.metric("Average Confidence", f"{stats['avg_confidence']*100:.1f}%")
    with col3:
        most_common = max(stats["class_distribution"], key=stats["class_distribution"].get) \
            if stats["class_distribution"] else "N/A"
        st.metric("Most Predicted", most_common)

    if stats["class_distribution"]:
        st.markdown("### Class Distribution")
        fig, ax = plt.subplots(figsize=(6, 3))
        color_map = {"Online": "#4C72B0", "Store": "#DD8452", "Hybrid": "#55A868"}
        dist = stats["class_distribution"]
        ax.bar(dist.keys(), dist.values(),
               color=[color_map.get(k, "#888") for k in dist.keys()], edgecolor="white")
        ax.set_ylabel("Count")
        ax.set_title("Prediction Class Distribution")
        for i, (k, v) in enumerate(dist.items()):
            ax.text(i, v + 0.1, str(v), ha='center', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("### Recent Predictions")
    rows = get_all_predictions(limit=50)
    if rows:
        display_df = pd.DataFrame([{
            "ID"       : r["id"],
            "Timestamp": r["timestamp"],
            "Prediction": r["predicted_class"],
            "Confidence": f"{r['confidence']*100:.1f}%",
            "Model"    : r["model_used"]
        } for r in rows])
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No predictions yet. Go to the Predict page to make your first prediction!")

    if st.button("🗑️ Clear All History", type="secondary"):
        delete_all_predictions()
        st.success("History cleared!")
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: ABOUT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "ℹ️ About":
    st.title("ℹ️ About This Application")
    st.markdown("""
    ## Predicting Consumer Shopping Channel Preference
    ### A Machine Learning Classification Framework

    ---
    ### 🎯 Business Problem
    An omni-channel retailer needs to identify whether customers prefer
    **Online**, **In-Store**, or **Hybrid** shopping to optimize marketing
    resource allocation and personalize the customer journey.

    ### 📊 Dataset
    - **Source:** [Kaggle – Online vs In-Store Shopping Behaviour](https://www.kaggle.com/datasets/shree0910/online-vs-in-store-shopping-behaviour-dataset)
    - **Records:** 11,789 customers
    - **Features:** 26 behavioral, demographic, and attitudinal variables

    ### 🤖 Models Evaluated
    | Model | Type | Accuracy |
    |-------|------|----------|
    | Logistic Regression ✅ | Human-selected | ~98.2% |
    | XGBoost | AI-recommended | ~96% |
    | SVM (RBF) | Benchmark | ~96.4% |
    | Random Forest | Benchmark | ~91% |
    | LightGBM | Benchmark | ~94% |

    ### 🔍 Key Features Influencing Prediction
    1. `monthly_online_orders` – Most significant digital behavior signal
    2. `avg_online_spend` – Spending pattern
    3. `tech_savvy_score` – Technology comfort
    4. `need_touch_feel_score` – Physical product preference
    5. `daily_internet_hours` – Digital engagement level

    ### 📁 Project Structure
    ```
    project/
    ├── ml-training/notebooks/   # Colab notebooks
    ├── models/                  # Trained model artifacts
    ├── py-app/                  # FastAPI REST API
    └── streamlit-app/           # This Streamlit UI (you are here)
    ```

    ### 👥 Team
    Built as a Machine Learning course project demonstrating end-to-end
    predictive analytics from data preprocessing to deployment.
    """)