import streamlit as st
import pandas as pd
import joblib

# ========================
# Load saved models/encoders
# ========================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
ohe = joblib.load("ohe.pkl")
le_trans = joblib.load("le_trans.pkl")   # Encoder for Transmission
le_sales = joblib.load("le_sales.pkl")   # Encoder for target (Sales Classification)

# ========================
# App Title
# ========================
st.title("BMW Sales Classification 🚗📊")
st.write("This app predicts **Sales Classification (High / Low)** based on car details.")

# ========================
# User Inputs
# ========================
model_name = st.selectbox("Select Model", 
                          ['5 Series','i8','X3','7 Series','M5','3 Series','X1','M3','X5','i3','X6'])

year = st.number_input("Production Year", min_value=2010, max_value=2025, value=2017)

region = st.selectbox("Region", ['Asia','North America','Middle East','South America','Europe','Africa'])

color = st.selectbox("Color", ['red','blue','black','silver','white','grey'])

fuel = st.selectbox("Fuel Type", ['Petrol','Hybrid','Diesel','Electric'])

trans = st.selectbox("Transmission", ['Manual','Automatic'])

engine = st.number_input("Engine Size (L)", min_value=1.5, max_value=5.0, value=3.0)

mileage = st.number_input("Mileage (KM)", min_value=0, max_value=500000, value=50000)

price = st.number_input("Price (USD)", min_value=3000, max_value=200000, value=75000)

sales_volume = st.number_input("Sales Volume", min_value=100, max_value=10000, value=5000)

# ========================
# Prepare Input Data
# ========================
input_data = pd.DataFrame({
    'Model':[model_name],
    'Year':[year],
    'Region':[region],
    'Color':[color],
    'Fuel_Type':[fuel],
    'Transmission':[trans],
    'Engine_Size_L':[engine],
    'Mileage_KM':[mileage],
    'Price_USD':[price],
    'Sales_Volume':[sales_volume]
})

# Feature Engineering
input_data["Car_Age"] = 2025 - input_data["Year"]
input_data.drop(columns=["Year"], inplace=True)

# Encode Transmission
input_data["Transmission_encoded"] = le_trans.transform(input_data["Transmission"])
input_data.drop(columns=["Transmission"], inplace=True)

# OneHotEncoding for categorical variables
cat_cols = ["Region", "Color", "Fuel_Type", "Model"]
encoded_array = ohe.transform(input_data[cat_cols])
encoded_df = pd.DataFrame(encoded_array, columns=ohe.get_feature_names_out(cat_cols), index=input_data.index)

final_input = pd.concat([input_data.drop(columns=cat_cols), encoded_df], axis=1)

# Scale numeric features
numeric_cols = ["Engine_Size_L", "Mileage_KM", "Price_USD", "Sales_Volume", "Car_Age"]
final_input[numeric_cols] = scaler.transform(final_input[numeric_cols])

# ========================
# Prediction
# ========================
if st.button("Predict Sales Classification"):
    prediction = model.predict(final_input)
    # Decode target back to original label (High / Low)
    result = le_sales.inverse_transform(prediction)[0]
    st.success(f"🚀 Prediction: {result} Sales")
