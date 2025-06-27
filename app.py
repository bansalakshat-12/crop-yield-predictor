import streamlit as st
import pickle
import numpy as np
import pandas as pd

# 🚀 Load model and features
with open("xgb_crop_yield_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("features_list.pkl", "rb") as f:
    feature_list = pickle.load(f)

# 🌾 Extract categories for dropdowns
states = sorted([col.replace("State_Name_", "") for col in feature_list if col.startswith("State_Name_")])
crops = sorted([col.replace("Crop_", "") for col in feature_list if col.startswith("Crop_")])
years = list(range(1997, 2016))  # Adjust to match your dataset

# 🖼️ Streamlit UI
st.title("🌾 Crop Yield Prediction App")
st.markdown("Select inputs to predict yield using a trained XGBoost model.")

# 🔘 Input selectors
selected_state = st.selectbox("Select State", ["-- Select State --"] + states)
selected_crop = st.selectbox("Select Crop", ["-- Select Crop --"] + crops)
selected_year = st.selectbox("Select Crop Year", ["-- Select Year --"] + years)

# 🧮 Run prediction only if inputs are selected and button is clicked
if st.button("Predict Yield"):

    # 🧪 Input validation
    if "-- Select" in (selected_state, selected_crop, str(selected_year)):
        st.error("Please select all inputs.")
    else:
        # Create zero-filled input matching feature list
        input_data = pd.DataFrame([np.zeros(len(feature_list))], columns=feature_list)
        input_data[f"State_Name_{selected_state}"] = 1
        input_data[f"Crop_{selected_crop}"] = 1
        input_data["Crop_Year"] = int(selected_year)

        # 🔮 Predict log yield and reverse transform
        log_pred = model.predict(input_data)[0]
        predicted_yield = np.expm1(log_pred)

        # 📈 Show result
        st.subheader("📊 Predicted Yield")
        st.success(f"Estimated Yield: **{predicted_yield:.2f} units**")
