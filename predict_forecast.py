import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model
import joblib
import os

def forecast_revenue(startup_name, n_years=6):
    """
    Forecast future revenue for a given startup using a pre-trained LSTM model.

    Parameters:
        startup_name (str): Name of the startup.
        n_years (int): Number of future years to forecast (default is 6).

    Returns:
        dict: Dictionary mapping future years to forecasted revenue values.
    """
    # Define file paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, "models", f"forecast_{startup_name}.h5")
    scaler_path = os.path.join(base_dir, "models", f"scaler_{startup_name}.pkl")
    csv_path = os.path.join(base_dir, "data", "startups_2025.csv")

    # Ensure the model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError("❌ Forecast model not found for the selected startup.")

    # Load pre-trained model and scaler
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    # Read revenue data
    df = pd.read_csv(csv_path)
    revenue_columns = [f"Revenue_{year}" for year in range(2019, 2025)]
    revenue_data = df[df['Startup Name'] == startup_name][revenue_columns].values.flatten()

    # Scale the revenue data
    revenue_scaled = scaler.transform(revenue_data.reshape(-1, 1)).flatten()

    # Use the last 3 years to start forecasting
    input_sequence = revenue_scaled[-3:].reshape(1, 3, 1)
    future_scaled_predictions = []

    # Generate forecasts iteratively
    for _ in range(n_years):
        next_pred = model.predict(input_sequence, verbose=0)[0][0]
        future_scaled_predictions.append(next_pred)

        # Update input sequence with the new prediction
        input_sequence = np.append(input_sequence.flatten()[1:], next_pred).reshape(1, 3, 1)

    # Convert predictions back to original scale
    future_revenue = scaler.inverse_transform(
        np.array(future_scaled_predictions).reshape(-1, 1)
    ).flatten()


    # Map each forecast to its corresponding year
    forecast_years = range(2025, 2025 + n_years)
    return dict(zip(forecast_years, future_revenue))

