import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def plot_revenue_chart(startup, df):
    # Extract company revenue data
    selected_company = df[df['Startup Name'] == startup]
    revenue_cols = [f'Revenue_{y}' for y in range(2019, 2025)]
    years_actual = np.array(range(2019, 2025))

    revenue_actual = pd.to_numeric(
        selected_company[revenue_cols].values.flatten(),
        errors='coerce'
    )

    # Basic validations
    if len(revenue_actual) == 0:
        st.warning("❌ No revenue data found for this startup.")
        return
    if np.isnan(revenue_actual).any():
        st.warning("❌ Missing or invalid revenue data. Try another startup.")
        return
    if np.count_nonzero(revenue_actual) == 0:
        st.warning("❌ All revenue values are zero. Forecast not possible.")
        return

    # Scale revenue
    scaler = MinMaxScaler()
    revenue_scaled = scaler.fit_transform(revenue_actual.reshape(-1, 1))

    # Prepare sequences (last 3 steps to predict next)
    X, y = [], []
    for i in range(3, len(revenue_scaled)):
        X.append(revenue_scaled[i - 3:i])
        y.append(revenue_scaled[i])
    X = np.array(X).reshape(-1, 3, 1)
    y = np.array(y)

    # Train LSTM model
    with st.spinner("🔄 Training LSTM model..."):
        model = Sequential()
        model.add(LSTM(50, activation='tanh', input_shape=(3, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=200, verbose=0)

    # Forecast next 6 years (2025–2030)
    input_seq = revenue_scaled[-3:].reshape(1, 3, 1)
    predicted_scaled = []
    for _ in range(6):
        pred = model.predict(input_seq, verbose=0)[0][0]
        predicted_scaled.append(pred)
        input_seq = np.append(input_seq.flatten()[1:], pred).reshape(1, 3, 1)

    # Inverse transform and clip
    revenue_forecast = scaler.inverse_transform(np.array(predicted_scaled).reshape(-1, 1)).flatten()
    future_years = np.arange(2025, 2031)

    #max_limit = 2 * revenue_actual.max()
    #revenue_forecast = np.clip(revenue_forecast, 0, max_limit)

    # Optional warning
    #if np.any(revenue_forecast > 1.5 * revenue_actual.max()):
        #st.warning("⚠️ Forecasted revenue may be unusually high. Consider retraining the model with more data.")

    # Plot chart
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=years_actual, y=revenue_actual, label='Actual Revenue', marker='o')
    sns.lineplot(x=future_years, y=revenue_forecast, label='Forecast (LSTM)', marker='o', color='orange')
    ax.set_title(f"📈 Forecasted Revenue for {startup}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Revenue (in Crores)")
    ax.set_ylim(0, max(max(revenue_actual), max(revenue_forecast)) * 1.2)
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

