"""Streamlit interface for the house price prediction demo."""
import streamlit as st
import matplotlib.pyplot as plt

from model import LOCATIONS, predict_price, train_model

st.set_page_config(page_title="House Price Prediction", layout="centered")


@st.cache_resource
def load_model():
    return train_model()


model, metrics, actual, predicted = load_model()
st.title("House Price Prediction")
st.write("Estimate a house price using area, bedrooms and location.")
st.caption("Machine learning demo · Linear Regression · Synthetic data (not market prices)")

with st.form("prediction"):
    left, right = st.columns(2)
    area = left.number_input("Area (m²)", min_value=40, max_value=250, value=100, step=5)
    bedrooms = right.number_input("Bedrooms", min_value=1, max_value=6, value=3, step=1)
    location = st.selectbox("Location", LOCATIONS)
    submitted = st.form_submit_button("Predict price", type="primary")

if submitted:
    price = predict_price(model, area, bedrooms, location)
    st.metric("Estimated house price", f"{price:,.2f} billion VND")

st.divider()
st.subheader("Model performance")
a, b, c = st.columns(3)
a.metric("R² score", f"{metrics['r2']:.3f}")
b.metric("MAE", f"{metrics['mae']:.3f}B VND")
c.metric("RMSE", f"{metrics['rmse']:.3f}B VND")
st.caption(f"Evaluated on {metrics['test_samples']} held-out synthetic samples.")
with st.expander("Actual vs. predicted prices"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(actual, predicted, alpha=0.65)
    low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    ax.plot([low, high], [low, high], "--", color="gray")
    ax.set(xlabel="Actual (billion VND)", ylabel="Predicted (billion VND)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
st.caption("For educational purposes only. Do not use this model for real estate valuation.")
