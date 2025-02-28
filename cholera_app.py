import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import folium
import streamlit as st
import requests
from streamlit_folium import folium_static

# Set random seed
np.random.seed(42)

# Generate synthetic data
dates = pd.date_range(start="2017-01-01", end="2024-12-01", freq="ME")
locations = ["Kanyama", "Matero", "Lusaka_District", "Chawama"]
rainfall = [np.random.uniform(0, 50) if month in [5, 6, 7, 8, 9] else np.random.uniform(150, 300) for month in dates.month]
temp = [np.random.uniform(15, 25) if month in [5, 6, 7, 8, 9] else np.random.uniform(25, 35) for month in dates.month]
cases = [np.random.randint(0, 100) if r < 100 and t < 20 else np.random.randint(200, 2000) for r, t in zip(rainfall, temp)]
deaths = [int(c * np.random.uniform(0.02, 0.05)) for c in cases]
pop_density = np.random.uniform(5000, 7000, len(dates))
sanitation = np.random.uniform(0.4, 0.6, len(dates))

# Add outbreak spikes
for i, date in enumerate(dates):
    if date.year == 2017 and date.month in [11, 12]:
        cases[i] = np.random.randint(1000, 2500)
    elif date.year == 2023 and date.month in [10, 11, 12]:
        cases[i] = np.random.randint(1000, 3000)
    elif date.year == 2024 and date.month in [1, 2, 3]:
        cases[i] = np.random.randint(500, 2000)

# Create DataFrame
data = pd.DataFrame({
    "Date": dates,
    "Cholera_Cases": cases,
    "Deaths": deaths,
    "Rainfall_mm": rainfall,
    "Temperature_C": temp,
    "Population_Density": pop_density,
    "Sanitation_Level": sanitation,
    "Location": np.random.choice(locations, len(dates))
})

# Train model
data["Outbreak"] = (data["Cholera_Cases"] > 200).astype(int)
X = data[["Rainfall_mm", "Temperature_C", "Population_Density", "Sanitation_Level"]]
y = data["Outbreak"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Fixed population density and sanitation per location
location_data = {
    "Kanyama": {"density": 6500, "sanitation": 0.5},
    "Matero": {"density": 6200, "sanitation": 0.45},
    "Lusaka_District": {"density": 5800, "sanitation": 0.55},
    "Chawama": {"density": 6700, "sanitation": 0.4}
}

# Fetch OpenWeatherMap current weather (today)
def get_current_weather():
    api_key = "60ca0403f7e3b2161c5b6ea63b642bee"
    url = f"http://api.openweathermap.org/data/2.5/weather?lat=-15.4167&lon=28.2833&appid={api_key}&units=metric"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        rain = data.get("rain", {}).get("1h", 0) * 10  # Convert 1h rain to mm, default 0 if no rain
        temp = data["main"]["temp"]  # °C
        return rain, temp
    except Exception as e:
        st.write(f"Current Weather API failed: {e}. Using defaults.")
        return 0, 25

# Fetch OpenWeatherMap forecast (tomorrow’s weather)
def get_weather_forecast():
    api_key = "60ca0403f7e3b2161c5b6ea63b642bee"
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat=-15.4167&lon=28.2833&appid={api_key}&units=metric"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        tomorrow = data["list"][8]  # Approx 24h ahead (3-hour steps, 8th entry)
        rain = tomorrow["rain"].get("3h", 0) * 10 if "rain" in tomorrow else 0  # Convert 3h rain to mm
        temp = tomorrow["main"]["temp"]  # °C
        return rain, temp
    except Exception as e:
        st.write(f"Weather API failed: {e}. Using defaults.")
        return 200, 30

# Streamlit app
st.title("Lusaka Cholera Prediction System")
st.write("Population density and sanitation are fixed per location; weather updates automatically from OpenWeatherMap.")

# Location selection
location_input = st.selectbox("Select Location", locations)
density_input = location_data[location_input]["density"]
sanitation_input = location_data[location_input]["sanitation"]
st.write(f"Fixed Values - Density: {density_input}, Sanitation: {sanitation_input}")

# Current conditions
st.subheader("Current Conditions (Today)")
current_rain, current_temp = get_current_weather()
st.write(f"Today’s Weather (OpenWeatherMap): Rainfall={current_rain:.2f}mm, Temp={current_temp:.2f}°C")
current_sample = pd.DataFrame([[current_rain, current_temp, density_input, sanitation_input]], 
                              columns=["Rainfall_mm", "Temperature_C", "Population_Density", "Sanitation_Level"])
current_raw_risk = model.predict_proba(current_sample)[0][1] * 100
current_risk = min(current_raw_risk, 50) if current_rain <= 150 else current_raw_risk
st.write(f"Current Outbreak Risk (Today): {current_risk:.2f}%")
if current_risk > 30:
    st.write(f"ALERT: High cholera risk in {location_input} today - {current_risk:.2f}%")
else:
    st.write(f"Low risk today: {current_risk:.2f}%")

# Tomorrow’s forecast
st.subheader("Forecast (Tomorrow)")
rainfall_auto, temp_auto = get_weather_forecast()
st.write(f"Tomorrow’s Weather (OpenWeatherMap): Rainfall={rainfall_auto:.2f}mm, Temp={temp_auto:.2f}°C")
sample = pd.DataFrame([[rainfall_auto, temp_auto, density_input, sanitation_input]], 
                      columns=["Rainfall_mm", "Temperature_C", "Population_Density", "Sanitation_Level"])
raw_risk = model.predict_proba(sample)[0][1] * 100
risk = min(raw_risk, 50) if rainfall_auto <= 150 else raw_risk
st.write(f"Predicted Outbreak Risk (Tomorrow): {risk:.2f}%")
if risk > 30:
    alert_msg = f"ALERT: High cholera risk in {location_input} tomorrow - {risk:.2f}%"
    st.write(alert_msg)
else:
    st.write(f"Low risk tomorrow: {risk:.2f}%")

# AI Analysis (Grok 3 simulation)
st.subheader("AI Analysis (Powered by Grok 3 Simulation)")
historical_risk = data.groupby("Location").apply(
    lambda x: (x["Cholera_Cases"] > 200).mean() * 100 if (x["Rainfall_mm"] > 150).any() else 0
).to_dict()
st.write("Historical Risk Trends (Synthetic Data):")
for loc, risk in historical_risk.items():
    st.write(f"{loc}: {risk:.2f}% risk when rainfall > 150mm")
st.write("Current Situation:")
if current_rain > 150:
    st.write(f"Today’s Rainfall={current_rain:.2f}mm exceeds 150mm threshold - High risk likely.")
else:
    st.write(f"Today’s Rainfall={current_rain:.2f}mm below 150mm - Lower risk expected.")
st.write("Tomorrow’s Forecast:")
if rainfall_auto > 150:
    st.write(f"Rainfall={rainfall_auto:.2f}mm exceeds 150mm threshold - High risk likely.")
else:
    st.write(f"Rainfall={rainfall_auto:.2f}mm below 150mm - Lower risk expected.")

# Risk map
st.subheader("Risk Map")
m = folium.Map(location=[-15.4167, 28.2833], zoom_start=12)
loc_coords = {
    "Kanyama": [-15.45, 28.25],
    "Matero": [-15.40, 28.20],
    "Lusaka_District": [-15.42, 28.28],
    "Chawama": [-15.43, 28.27]
}
for loc, coords in loc_coords.items():
    loc_density = location_data[loc]["density"]
    loc_sanitation = location_data[loc]["sanitation"]
    loc_sample = pd.DataFrame([[rainfall_auto, temp_auto, loc_density, loc_sanitation]], 
                              columns=["Rainfall_mm", "Temperature_C", "Population_Density", "Sanitation_Level"])
    loc_risk = model.predict_proba(loc_sample)[0][1] * 100
    loc_risk = min(loc_risk, 50) if rainfall_auto <= 150 else loc_risk
    folium.Marker(coords, popup=f"{loc}: {loc_risk:.2f}%").add_to(m)
folium_static(m)
