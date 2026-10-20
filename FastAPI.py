import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


# ---------------------------------------------------------
# Load model + preprocessing artifacts
# ---------------------------------------------------------

model = joblib.load("car_price_model.pkl")
encoders = joblib.load("encoders.pkl")
feature_columns = joblib.load("feature_columns.pkl")

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
        "Model": data.Model
    }

    # Encode categorical columns
    for col in encoders:
        user_input[col] = encoders[col].transform(
            [user_input[col]]
        )[0]

    # Correct feature order
    input_df = pd.DataFrame(
        [user_input]
    )[feature_columns]

    # Prediction
    log_price_pred = model.predict(input_df)[0]

    price_pred = np.exp(log_price_pred)

    price_pred_inr = price_pred * USD_TO_INR

    return {
        "predicted_price_inr": round(
            float(price_pred_inr), 2
        )
    }