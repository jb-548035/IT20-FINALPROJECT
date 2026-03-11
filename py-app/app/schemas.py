"""
schemas.py
----------
Pydantic schemas define the shape of data the API receives (request body)
and the shape of data the API returns (response body).

Think of schemas as contracts: they validate incoming data automatically
and generate OpenAPI documentation for free.
"""

from pydantic import BaseModel, Field, validator
from typing import Literal, Optional


# ─── INPUT SCHEMA ─────────────────────────────────────────────────────────────
# This is the structure of the JSON body the client must send to /predict

class CustomerFeatures(BaseModel):
    """
    All 24 raw input features needed to predict a customer's
    shopping channel preference (Online / Store / Hybrid).

    Engineered features (ratios, composites) are computed inside
    model_loader.py — the client does NOT need to send them.
    """

    # ── Demographic ────────────────────────────────────────────────────────────
    Age: float = Field(..., ge=10, le=100, description="Customer age in years")
    monthly_income: float = Field(..., ge=0, description="Monthly income in INR")
    gender: Literal["Male", "Female", "Other"] = Field(..., description="Customer gender")
    city_tier: Literal["Tier 1", "Tier 2", "Tier 3"] = Field(
        ..., description="City classification (Tier 1 = metro, Tier 3 = small city)"
    )

    # ── Digital Behavior ───────────────────────────────────────────────────────
    daily_internet_hours: float = Field(..., ge=0, le=24, description="Hours online per day")
    smartphone_usage_years: float = Field(..., ge=0, description="Years of smartphone usage")
    social_media_hours: float = Field(..., ge=0, le=24, description="Daily social media hours")
    online_payment_trust_score: float = Field(..., ge=1, le=10, description="Trust in digital payments (1–10)")
    tech_savvy_score: float = Field(..., ge=1, le=10, description="Comfort with technology (1–10)")
    monthly_online_orders: float = Field(..., ge=0, description="Online orders per month")

    # ── Shopping Behavior ──────────────────────────────────────────────────────
    monthly_store_visits: float = Field(..., ge=0, description="Store visits per month")
    avg_online_spend: float = Field(..., ge=0, description="Average online purchase value in INR")
    avg_store_spend: float = Field(..., ge=0, description="Average in-store purchase value in INR")
    discount_sensitivity: float = Field(..., ge=1, le=10, description="Sensitivity to discounts (1–10)")
    return_frequency: float = Field(..., ge=1, le=10, description="Product return frequency (1–10)")
    avg_delivery_days: float = Field(..., ge=0, description="Average delivery time in days")
    delivery_fee_sensitivity: float = Field(..., ge=1, le=10, description="Sensitivity to delivery fees (1–10)")

    # ── Attitudinal ────────────────────────────────────────────────────────────
    free_return_importance: float = Field(..., ge=1, le=10, description="Importance of free returns (1–10)")
    product_availability_online: float = Field(..., ge=1, le=10, description="Perceived online availability (1–10)")
    impulse_buying_score: float = Field(..., ge=1, le=10, description="Impulse buying likelihood (1–10)")
    need_touch_feel_score: float = Field(..., ge=1, le=10, description="Preference to touch products (1–10)")
    brand_loyalty_score: float = Field(..., ge=1, le=10, description="Brand loyalty level (1–10)")
    environmental_awareness: float = Field(..., ge=1, le=10, description="Eco-consciousness (1–10)")
    time_pressure_level: float = Field(..., ge=1, le=10, description="Perceived time pressure (1–10)")

    class Config:
        # Show an example payload in the Swagger UI docs
        schema_extra = {
            "example": {
                "Age": 28,
                "monthly_income": 45000,
                "gender": "Male",
                "city_tier": "Tier 1",
                "daily_internet_hours": 5.5,
                "smartphone_usage_years": 7,
                "social_media_hours": 2.5,
                "online_payment_trust_score": 8,
                "tech_savvy_score": 7,
                "monthly_online_orders": 12,
                "monthly_store_visits": 2,
                "avg_online_spend": 3200,
                "avg_store_spend": 800,
                "discount_sensitivity": 7,
                "return_frequency": 4,
                "avg_delivery_days": 3,
                "delivery_fee_sensitivity": 6,
                "free_return_importance": 8,
                "product_availability_online": 8,
                "impulse_buying_score": 6,
                "need_touch_feel_score": 3,
                "brand_loyalty_score": 5,
                "environmental_awareness": 7,
                "time_pressure_level": 8
            }
        }


# ─── OUTPUT SCHEMA ────────────────────────────────────────────────────────────
# This is the structure of the JSON the API returns after prediction

class PredictionResponse(BaseModel):
    """
    API response containing the predicted shopping preference
    and the probability distribution across all three classes.
    """

    predicted_class: Literal["Online", "Store", "Hybrid"] = Field(
        ..., description="Predicted shopping preference"
    )
    confidence: float = Field(
        ..., ge=0, le=1,
        description="Confidence score of the predicted class (0–1)"
    )
    probabilities: dict = Field(
        ...,
        description="Probability for each class: {Online: %, Store: %, Hybrid: %}"
    )
    model_used: str = Field(
        ..., description="Name of the model that produced the prediction"
    )

    class Config:
        schema_extra = {
            "example": {
                "predicted_class": "Online",
                "confidence": 0.921,
                "probabilities": {
                    "Hybrid": 0.021,
                    "Online": 0.921,
                    "Store": 0.058
                },
                "model_used": "Logistic Regression (Tuned)"
            }
        }


# ─── BATCH INPUT SCHEMA ───────────────────────────────────────────────────────

class BatchPredictionRequest(BaseModel):
    """
    Accepts a list of customer records for batch prediction.
    Maximum 100 records per request.
    """
    customers: list[CustomerFeatures] = Field(
        ..., min_items=1, max_items=100,
        description="List of customer feature records (max 100)"
    )


class BatchPredictionResponse(BaseModel):
    """
    Batch prediction results with per-customer predictions.
    """
    total: int = Field(..., description="Total number of predictions made")
    predictions: list[PredictionResponse]