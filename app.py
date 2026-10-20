import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

# ---------------------------------------------------------
# Load saved model + preprocessing artifacts
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("car_price_model.pkl")
    encoders = joblib.load("encoders.pkl")
    cat_cols = joblib.load("cat_cols.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    numeric_ranges = joblib.load("numeric_ranges.pkl")
    return model, encoders, cat_cols, feature_columns, numeric_ranges

model, encoders, cat_cols, feature_columns, numeric_ranges = load_artifacts()

# Approx conversion rate from the dataset's original currency to INR.
# Adjust this number if you want to match the current exchange rate.
USD_TO_INR = 90

st.title("🚗 Car Price Predictor")
st.write("Fill in the car details below and get an estimated price.")

# ---------------------------------------------------------
# Build the input form dynamically from saved columns
# ---------------------------------------------------------
user_input = {}

with st.form("car_form"):
    for col in feature_columns:
        if col in cat_cols:
            # Dropdown using the original text categories the encoder learned
            options = list(encoders[col].classes_)
            choice = st.selectbox(col, options)
            user_input[col] = encoders[col].transform([choice])[0]
        else:
            lo, hi = numeric_ranges.get(col, (0.0, 100.0))
            default = float((lo + hi) / 2)
            if col == "Year":
                # Year should always be a whole number, not a decimal
                user_input[col] = st.number_input(
                    col, min_value=int(lo), max_value=int(hi), value=int(default),
                    step=1
                )
            else:
                user_input[col] = st.number_input(
                    col, min_value=float(lo), max_value=float(hi), value=default
                )

    submitted = st.form_submit_button("Predict Price")

# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------
if submitted:
    input_df = pd.DataFrame([user_input])[feature_columns]
    log_price_pred = model.predict(input_df)[0]
    price_pred = np.exp(log_price_pred)  # undo the log transform from training
    price_pred_inr = price_pred * USD_TO_INR

    st.success(f"### Estimated Price: ₹{price_pred_inr:,.2f}")
    st.caption("Prediction based on a regression model trained on used-car sales data.")