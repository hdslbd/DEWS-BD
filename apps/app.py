import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# Set page configuration
st.set_page_config(page_title="Dengue Early Warning System for Bangladesh(DEWS-BD)",
                   layout="wide",
                   page_icon="🧑‍⚕️")


# Define a list of districts
bangladesh_districts = [
    'Dhaka', 'Chittagong', 'Khulna', 'Sylhet', 'Rajshahi',
    'Barisal', 'Comilla', 'Rangpur', 'Mymensingh', 'Narayanganj',
    'Gazipur', 'Jessore', 'Dinajpur', 'Bogra', 'Pabna'
]

# Load or simulate data
np.random.seed(42)
dates = pd.date_range(start='2007-01-01', end='2024-01-01', freq='M')
actual_data = np.random.poisson(lam=500, size=len(dates)) + np.random.normal(scale=100, size=len(dates))
predicted_data = actual_data + np.random.normal(scale=50, size=len(dates))

data = pd.DataFrame({
    'Date': dates,
    'Actual': actual_data,
    'Predicted': predicted_data
})

# App title
st.title('Dengue Early Warning System for Bangladesh(DEWS-BD)')

# Sidebar for district selection
st.sidebar.title("DEWS-BD")  # Sidebar title
district = st.sidebar.selectbox('Select District:', bangladesh_districts)

# Filter data based on district if your data supports it; here it's static for demo
# Assuming data doesn't actually vary by district in this example

# Line chart using Plotly
fig = px.line(data, x='Date', y=['Actual', 'Predicted'], title=f'Dengue Patient Count Forecast for {district}')
st.plotly_chart(fig)

# Model evaluation
rmse = np.sqrt(mean_squared_error(data['Actual'], data['Predicted']))
st.write(f"Root Mean Square Error (RMSE): {rmse:.2f}")

# Prediction for a specific month
month_to_predict = st.sidebar.selectbox('Select Month:', data['Date'].dt.month_name().unique())
filtered_data = data[data['Date'].dt.month_name() == month_to_predict]
prediction = filtered_data['Predicted'].mean()
st.sidebar.write(f'Prediction for {month_to_predict}: {int(prediction)}')

# To run this app, save the code in a file app.py and run `streamlit run app.py`
