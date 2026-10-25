import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(title="Car Price API")


# ---------------------------------------------------------
# Load model + preprocessing artifacts
# ---------------------------------------------------------
model = joblib.load("car_price_model.pkl")
encoders = joblib.load("encoders.pkl")
cat_cols = joblib.load("cat_cols.pkl")
feature_columns = joblib.load("feature_columns.pkl")
numeric_ranges = joblib.load("numeric_ranges.pkl")

USD_TO_INR = 90


# ---------------------------------------------------------
# Input schema
# ---------------------------------------------------------
class CarInput(BaseModel):
    Brand: str
    Body: str
    Mileage: float
    EngineV: float
    Engine_Type: str
    Registration: str
    Year: int
    Model: str


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------
@app.get("/")
def home():
    return {"status": "ok"}


# ---------------------------------------------------------
# Options for the Streamlit form (dropdown values + ranges)
# ---------------------------------------------------------
@app.get("/options")
def options():
    categories = {
        col: [str(c) for c in encoders[col].classes_] for col in cat_cols
    }
    ranges = {
        col: [float(lo), float(hi)] for col, (lo, hi) in numeric_ranges.items()
    }
    return {
        "feature_columns": list(feature_columns),
        "cat_cols": list(cat_cols),
        "categories": categories,
        "numeric_ranges": ranges,
    }


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
@app.post("/predict")
def predict(data: CarInput):
    user_input = {
        "Brand": data.Brand,
        "Body": data.Body,
        "Mileage": data.Mileage,
        "EngineV": data.EngineV,
        "Engine Type": data.Engine_Type,
        "Registration": data.Registration,
        "Year": data.Year,
        "Model": data.Model,
    }

    # Encode categorical columns
    for col in encoders:
        try:
            user_input[col] = encoders[col].transform([user_input[col]])[0]
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"'{user_input[col]}' is not a known value for {col}",
            )

    # Correct feature order
    input_df = pd.DataFrame([user_input])[feature_columns]

    # Prediction (model was trained on log price)
    log_price_pred = model.predict(input_df)[0]
    price_pred = np.exp(log_price_pred)
    price_pred_inr = price_pred * USD_TO_INR

    return {"predicted_price_inr": round(float(price_pred_inr), 2)}