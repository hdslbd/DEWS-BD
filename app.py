import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

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

# Normalize data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data[['Actual']].values.reshape(-1, 1))

# Prepare data for LSTM
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 1
trainX, trainY = create_dataset(data_scaled, look_back)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

# LSTM Model Initialization
def train_lstm(trainX, trainY):
    model = Sequential()
    model.add(LSTM(50, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=0)
    return model

lstm_model = train_lstm(trainX, trainY)

# Streamlit UI
st.title('Dengue Outbreak Prediction in Bangladesh')

# Sidebar for district and model selection
district = st.sidebar.selectbox('Select District:', bangladesh_districts)
model_choice = st.sidebar.radio("Choose the model:", ('SARIMAX', 'LSTM', 'Holt-Winters'))

# Month selection for prediction
selected_month = st.sidebar.selectbox('Select Month:', [m for m in pd.date_range(start='2024-01-01', periods=12, freq='M').month_name()])

# Display line chart
fig = px.line(data, x='Date', y=['Actual', 'Predicted'], title=f'Dengue Patient Count Forecast for {district}')
st.plotly_chart(fig)

# Prediction logic based on model choice
if st.button('Show Predictions'):
    if model_choice == 'SARIMAX':
        model = SARIMAX(data['Actual'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit(disp=False)
        predictions = model_fit.forecast(steps=12)
    elif model_choice == 'LSTM':
        last_values = np.array([data_scaled[-1]])
        last_values = last_values.reshape(1, 1, 1)
        predictions = lstm_model.predict(last_values)
        predictions = scaler.inverse_transform(predictions)
    elif model_choice == 'Holt-Winters':
        model = ExponentialSmoothing(data['Actual'], seasonal='add', seasonal_periods=12)
        model_fit = model.fit()
        predictions = model_fit.forecast(steps=12)
    
    # Filter predictions for the selected month
    month_index = pd.date_range(start='2024-01-01', periods=12, freq='M').month_name().tolist().index(selected_month)
    selected_prediction = predictions[month_index]

    st.write(f"Forecast for {selected_month} using {model_choice}: {selected_prediction}")

# Display RMSE for evaluation
if model_choice != 'LSTM':  # LSTM needs different handling for RMSE due to scaling
    rmse = np.sqrt(mean_squared_error(data['Actual'], data['Predicted']))
    st.write(f"Root Mean Square Error (RMSE): {rmse:.2f}")
