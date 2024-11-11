import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load the dataset
@st.cache
def load_data():
    df = pd.read_csv(r"C:\Users\hp\OneDrive\bengaluru\bengaluru.csv", index_col='date_time')
    df.index = pd.to_datetime(df.index, format='%d-%m-%Y %H:%M')
    return df

# Load the second DataFrame (for example, 2019 data)
@st.cache
def load_data_2019():
    df2 = pd.read_csv(r"bengaluru2019.csv", index_col='date_time')
    df2.index = pd.to_datetime(df2.index, format='%d-%m-%Y %H:%M')
    return df2

# Forecasting function using ARIMA
def forecast_arima(data, order, periods):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=periods)
    return forecast

# Main function to run the app
def main():
    st.title("Weather Data Forecasting with ARIMA")

    # Load data
    df = load_data()
    df2 = load_data_2019()

    # Check the shape of the DataFrame
    st.write("Data shape:", df.shape)

    # Filter data for the last month of December 2018
    start_date = '2018-12-22 00:00:00'
    end_date = '2018-12-31 23:00:00'
    filtered_data = df[start_date:end_date]

    st.write("### Historical Weather Data (Last Month of December 2018)")
    st.line_chart(filtered_data['tempC'])  # Display temperature data for December 2018

    # User input for ARIMA parameters
    p = st.number_input("Select p (AR term):", min_value=0, max_value=5, value=1)
    d = st.number_input("Select d (I term):", min_value=0, max_value=2, value=1)
    q = st.number_input("Select q (MA term):", min_value=0, max_value=5, value=1)
    periods = st.number_input("Enter number of periods to forecast:", min_value=1, max_value=30, value=12)

    # Forecasting
    if st.button("Forecast"):
        with st.spinner("Forecasting..."):
            forecast = forecast_arima(filtered_data['tempC'], order=(p, d, q), periods=periods)
            forecast_index = pd.date_range(start=filtered_data.index[-1] + pd.Timedelta(hours=1), periods=periods, freq='H')
            forecast_series = pd.Series(forecast, index=forecast_index)

            # Plotting the forecast
            plt.figure(figsize=(10, 5))
            plt.plot(filtered_data['tempC'], label='Historical Data (Dec 2018)', color='blue')
            plt.plot(forecast_series, label='Forecast', color='orange')
            plt.title('Temperature Forecast for December 2018')
            plt.xlabel('Date')
            plt.ylabel('Temperature (°C)')
            plt.legend()
            st.pyplot(plt)

            # Plotting Bengaluru 2018 and 2019 data
            st.write("### Historical Weather Data for Bengaluru (2018 and 2019)")
            plt.figure(figsize=(25, 15))
            plt.plot(df['2018-12-22 00:00:00':'2018-12-31 23:00:00 ']['tempC'], label='Previous Data', color='orange')
            plt.plot(df2['2019-01-01 00:00:00':'2019-01-02 23:00:00']['tempC'], label='Present Data', color='blue')
            plt.title('Original Data')
            plt.xlabel('Date')
            plt.ylabel('Temperature (°C)')
            plt.legend()
            st.pyplot(plt)

# Run the app
if __name__ == '__main__':
    main()