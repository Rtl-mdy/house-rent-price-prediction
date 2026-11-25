from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "rent_model_uci.joblib"
DATA_PATH = ROOT / "data" / "cleaned_apartment_rent.csv"

st.set_page_config(page_title="House Rent Price Prediction", page_icon="🏠", layout="centered")
st.title("House Rent Price Prediction")
st.caption("Academic demonstration using US apartment listings; estimates are in USD/month.")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python train_model_uci.py` first.")
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_locations():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Cleaned dataset not found. Run `python train_model_uci.py` first.")
    data = pd.read_csv(DATA_PATH)
    states = sorted(data["state"].dropna().unique().tolist())
    mapping = {s: sorted(data.loc[data["state"] == s, "cityname"].dropna().unique().tolist())
               for s in states}
    return states, mapping

try:
    model = load_model()
    states, state_cities = load_locations()
except Exception as e:
    st.error(str(e))
    st.stop()

with st.form("rent_form"):
    state = st.selectbox("State", states, index=0)
    city = st.selectbox("City", state_cities[state])
    col1, col2 = st.columns(2)
    with col1:
        bedrooms = st.number_input("Bedrooms (0 = studio)", min_value=0, max_value=5, value=2, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=4.0, value=1.5, step=0.5)
    with col2:
        square_feet = st.number_input("Area (square feet)", min_value=100, max_value=10000, value=900, step=50)
    submitted = st.form_submit_button("Predict monthly rent")

if submitted:
    row = pd.DataFrame([{
        "bedrooms": bedrooms, "bathrooms": bathrooms, "square_feet": square_feet,
        "cityname": city, "state": state
    }])
    estimate = max(0.0, float(model.predict(row)[0]))
    st.success(f"Estimated monthly rent: ${estimate:,.2f} USD")
    st.caption("This is a model estimate, not a listing quote or financial advice.")
