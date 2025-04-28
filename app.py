%%writefile app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import random
from statsmodels.tsa.arima.model import ARIMA

# --- Constants ---
API_KEY = "01851383408244ff9f8112646252704"  # Replace with your real WeatherAPI key
CURRENT_URL = "http://api.weatherapi.com/v1/current.json"
FORECAST_URL = "http://api.weatherapi.com/v1/forecast.json"

# --- Load crop data (CSV) ---
crop_data = pd.read_csv('crop_data.csv')  # Ensure you have crop_data.csv loaded

# --- Functions ---

def get_current_weather(city):
    params = {
        "key": API_KEY,
        "q": city,
        "aqi": "no"
    }
    response = requests.get(CURRENT_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def get_forecast_weather(city):
    params = {
        "key": API_KEY,
        "q": city,
        "days": 7,
        "aqi": "no",
        "alerts": "no"
    }
    response = requests.get(FORECAST_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def simulate_monthly_rainfall_forecast(start_date, months=3):
    months_list = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    start_month = start_date.month
    forecast = pd.DataFrame({
        "month": [months_list[(start_month + i) % 12] for i in range(months)],
        "rainfall_mm": [random.randint(50, 200) for _ in range(months)],
    })
    return forecast

def prepare_yield_data():
    crop_data_filtered = crop_data[['Annual_Rainfall', 'Yield']].dropna()
    return crop_data_filtered

def train_yield_model():
    data = prepare_yield_data()
    if data.empty:
        return None
    X = data[['Annual_Rainfall']]
    y = data['Yield']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict_yield(model, forecasted_rainfall):
    if model:
        predicted_yield = model.predict([[forecasted_rainfall]])
        return predicted_yield[0]
    return None

def generate_short_summary(season, state, acres, forecasted_rainfall, selected_date, model):
    crop_info = crop_data[(crop_data['Season'] == season) & (crop_data['State'] == state)]
    if crop_info.empty:
        return f"Error: No data available for season {season} and state {state}."

    forecasted_rainfall_avg = forecasted_rainfall['rainfall_mm'].mean()
    predicted_yield = predict_yield(model, forecasted_rainfall_avg)

    summary = f"🌱 Cultivation Recommendation for {season} Season ({selected_date.strftime('%Y-%m-%d')}):\n"
    if predicted_yield is not None:
        summary += f"\n📊 Predicted average yield based on rainfall: {predicted_yield:.2f} tons.\n"
    else:
        summary += "\n⚠️ Crop yield prediction not available.\n"

    higher_production_crop = crop_info.loc[crop_info['Production'].idxmax()]
    lower_production_crop = crop_info.loc[crop_info['Production'].idxmin()]

    summary += f"\n🔝 **Highest Production Crop:** {higher_production_crop['Crop']} ({higher_production_crop['Production']} tons/acre).\n"
    summary += f"🔻 **Lowest Production Crop:** {lower_production_crop['Crop']} ({lower_production_crop['Production']} tons/acre).\n"
    summary += f"\n⚡ Avg Rainfall Next 3 Months: {forecasted_rainfall_avg:.2f} mm\n"

    if forecasted_rainfall_avg < 60:
        summary += "\n💧 Recommendation: Low rainfall. Grow drought-resistant crops."
    elif forecasted_rainfall_avg > 120:
        summary += "\n🌧️ Recommendation: High rainfall. Focus on water-tolerant crops."

    if predicted_yield is not None and higher_production_crop['Yield'] > 0:
        recommended_acres = (acres * predicted_yield) / higher_production_crop['Yield']
        summary += f"\n\n🚜 For {acres} acres, {higher_production_crop['Crop']} could be a good focus."
    else:
        summary += "\n\n⚠️ Acreage recommendation unavailable."

    summary += "\n\n🌿 Stay updated with weather for better decisions."
    return summary

def arima_forecasting(rainfall_data, months=3):
    if len(rainfall_data) < 2:
        return np.array([np.nan] * months)
    try:
        model = ARIMA(rainfall_data, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=months)
        return forecast
    except Exception as e:
        print(f"ARIMA Error: {e}")
        return np.array([np.nan] * months)

# --- Streamlit UI ---
st.set_page_config(page_title="🌤️ Agritech Analytics and Cultivation Recommendation System", layout="wide")

# --- CSS Styling ---
st.markdown(
    """
    <style>
    .main {
        background-color: #f7f7f7;
    }
    section[data-testid="stSidebar"] {
        background-color: #95a5a6;
    }
    h1 {
        color: #2E7D32;
    }
    h2 {
        color: #4d5656;
    }
    .stButton>button {
        background-color: #2E7D32;
        color: white;
        font-weight: bold;
    }
    .stTextInput>div>div>input, .stSelectbox>div>div>select, .stNumberInput>div>div>input, .stDateInput>div>div>input {
        font-size: 16px;
        border: 1px solid #ccc;
        border-radius: 4px;
        padding: 0.5em;
    }
    .report-box {
        border: 1px solid #ccc;
        padding: 15px;
        border-radius: 5px;
        background-color: #f9f9f9;
    }
    .metric-box {
        border: 1px solid #ddd;
        padding: 10px;
        border-radius: 5px;
        background-color: #fff;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌤️ Agritech Analytics and Cultivation Recommendation System")

# --- Sidebar for Selections ---
with st.sidebar:
    st.header("⚙️ Crop and Date Selection")
    location_input = st.text_input("Enter City for Weather", "Hyderabad", key="city")
    seasons = crop_data['Season'].unique()
    states = crop_data['State'].unique()

    st.selectbox("Select Season", seasons, key="season")
    st.selectbox("Select State", states, key="state")
    st.number_input("Enter Cultivation Area (Acres)", min_value=1, key="acres")
    st.date_input("Select Start Date", min_value=datetime.today(), key="start_date")
    search_button = st.button("🔍 Get Forecast & Recommendations")

# --- Main Area ---
if search_button:
    location_input = st.session_state.get("city", "Hyderabad")
    season_selected = st.session_state.get("season", seasons[0] if seasons.any() else "")
    state_selected = st.session_state.get("state", states[0] if states.any() else "")
    acres = st.session_state.get("acres", 1)
    selected_date = st.session_state.get("start_date", datetime.today())

    if location_input:
        current_weather = get_current_weather(location_input)
        forecast_weather = get_forecast_weather(location_input)

        st.subheader(f"📍 Weather Report for {location_input}")
        if current_weather:
            col1, col2, col3, col4 = st.columns(4)
            col1.markdown(f"<div class='metric-box'>🌡️ Temperature<br><h2>{current_weather['current']['temp_c']}°C</h2></div>", unsafe_allow_html=True)
            col2.markdown(f"<div class='metric-box'>💧 Humidity<br><h2>{current_weather['current']['humidity']}%</h2></div>", unsafe_allow_html=True)
            col3.markdown(f"<div class='metric-box'>💨 Wind Speed<br><h2>{current_weather['current']['wind_kph']} km/h</h2></div>", unsafe_allow_html=True)
            col4.markdown(f"<div class='metric-box'>☀️ UV Index<br><h2>{current_weather['current']['uv']}</h2></div>", unsafe_allow_html=True)

            st.subheader("7-Day Temperature Forecast")
            forecast_data = forecast_weather['forecast']['forecastday']
            forecast_dates = [data['date'] for data in forecast_data]
            forecast_temps = [data['day']['avgtemp_c'] for data in forecast_data]

            fig_weather = go.Figure()
            fig_weather.add_trace(go.Scatter(x=forecast_dates, y=forecast_temps, mode='lines+markers', name='Temperature (°C)'))
            fig_weather.update_layout(title="7-Day Temperature Forecast", xaxis_title="Date", yaxis_title="Temperature (°C)", template="plotly_dark")
            st.plotly_chart(fig_weather)

            rainfall_forecast_df = simulate_monthly_rainfall_forecast(selected_date, months=3)

            st.subheader("3-Month Rainfall Forecast")
            fig_rainfall = go.Figure()
            fig_rainfall.add_trace(go.Bar(x=rainfall_forecast_df['month'], y=rainfall_forecast_df['rainfall_mm'], name='Rainfall (mm)'))
            fig_rainfall.update_layout(title="Monthly Rainfall Forecast", xaxis_title="Month", yaxis_title="Rainfall (mm)", template="plotly_dark")
            st.plotly_chart(fig_rainfall)

            model = train_yield_model()

            st.subheader("🌱 Crop Recommendation & Analysis")
            summary = generate_short_summary(season_selected, state_selected, acres, rainfall_forecast_df, selected_date, model)
            st.markdown(f"<div class='report-box'><h3>Recommendation Summary</h3><p>{summary}</p></div>", unsafe_allow_html=True)

            st.subheader("📊 Crop Production Estimates")
            crops = crop_data[(crop_data['Season'] == season_selected) & (crop_data['State'] == state_selected)].sort_values(by='Production', ascending=False)
            if not crops.empty:
                fig_production = go.Figure()
                fig_production.add_trace(go.Bar(x=crops['Crop'], y=crops['Production'], name='Production (tons/acre)'))
                fig_production.update_layout(title=f"Crop Production Estimates for {season_selected} in {state_selected}", xaxis_title="Crop", yaxis_title="Production (tons/acre)", template="plotly_dark")
                st.plotly_chart(fig_production)
            else:
                st.warning(f"No crop production data available for {season_selected} in {state_selected}.")
        else:
            st.error("Error fetching current weather data.")
        if not forecast_weather:
            st.error("Error fetching forecast data.")
    else:
        st.warning("Please enter a city to proceed.")
