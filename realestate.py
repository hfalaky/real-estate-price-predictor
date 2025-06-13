import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Paris Price Estimator", page_icon="🏠")

st.title("🏠 Paris Apartment Price Estimator")

try:
    # Load the trained model
    model = joblib.load("price_model.pkl")

    # --- Input fields ---
    area = st.number_input("Surface area (m²)", min_value=10, max_value=300, value=50)
    rooms = st.slider("Number of rooms", min_value=1, max_value=10, value=2)
    arrondissement = st.selectbox("Arrondissement", list(range(1, 21)))

    # --- Predict button ---
    if st.button("Predict Price (€)"):
        input_data = np.array([[area, rooms, arrondissement]])
        prediction = model.predict(input_data)[0]

        st.success(f"🏷️ Estimated price: **€{int(prediction):,}**")

except FileNotFoundError:
    st.error("❌ Model file not found. Make sure 'price_model.pkl' exists.")
except Exception as e:
    st.error("⚠️ An unexpected error occurred:")
    st.code(str(e))
