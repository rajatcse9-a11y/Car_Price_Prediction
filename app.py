import os

import requests
import streamlit as st

# Local: FastAPI same laptop pe chal rahi hoti hai.
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

st.title("🚗 Car Price Predictor")
st.write("Fill in the car details below and get an estimated price.")


# ---------------------------------------------------------
# Form options come from the FastAPI backend
# ---------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_options():
    # Timeout bada rakha hai, kyunki free tier pe API so jaati hai
    response = requests.get(f"{API_URL}/options", timeout=90)
    response.raise_for_status()
    return response.json()


try:
    with st.spinner("Connecting to the prediction API..."):
        options = get_options()
except requests.RequestException as e:
    st.error(f"API se connect nahi ho paya ({API_URL}). Error: {e}")
    st.stop()

feature_columns = options["feature_columns"]
cat_cols = options["cat_cols"]
categories = options["categories"]
numeric_ranges = options["numeric_ranges"]


# ---------------------------------------------------------
# Build the form dynamically
# ---------------------------------------------------------
payload = {}

with st.form("car_form"):
    for col in feature_columns:
        # API ke fields mein space nahi hota ("Engine Type" -> "Engine_Type")
        key = col.replace(" ", "_")

        if col in cat_cols:
            payload[key] = st.selectbox(col, categories[col])
        else:
            lo, hi = numeric_ranges.get(col, [0.0, 100.0])
            default = (lo + hi) / 2
            if col == "Year":
                payload[key] = st.number_input(
                    col, min_value=int(lo), max_value=int(hi),
                    value=int(default), step=1,
                )
            else:
                payload[key] = st.number_input(
                    col, min_value=float(lo), max_value=float(hi),
                    value=float(default),
                )

    submitted = st.form_submit_button("Predict Price")


# ---------------------------------------------------------
# Predict via the API
# ---------------------------------------------------------
if submitted:
    try:
        with st.spinner("Predicting..."):
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=90)

        if response.status_code == 200:
            price = response.json()["predicted_price_inr"]
            st.success(f"### Estimated Price: ₹{price:,.2f}")
            st.caption("Prediction based on a regression model trained on used-car sales data.")
        else:
            st.error(f"API error {response.status_code}: {response.text}")

    except requests.RequestException as e:
        st.error(f"Request failed: {e}")
