import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib
import os

base_dir = os.path.dirname(os.path.dirname(__file__))
csv_path = os.path.join(base_dir, "data", "startups_2025.csv")
df = pd.read_csv(csv_path)
startups = df['Startup Name'].unique()

os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)

for startup_name in startups:
    revenue_cols = [f"Revenue_{y}" for y in range(2019, 2025)]
    revenues = df.loc[df['Startup Name'] == startup_name, revenue_cols].values.flatten()

    # Skip if insufficient data or NaNs
    if len(revenues) < 6 or np.isnan(revenues).any():
        continue

    # Train on first 4 years (2019-2022), test on last 2 years (2023-2024)
    train_revenues = revenues[:4]
    test_revenues = revenues[4:]

    # Scale based only on training data
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_revenues.reshape(-1, 1))

    # Prepare train sequences: input = revenue(t), output = revenue(t+1)
    X_train = train_scaled[:-1].reshape(-1, 1, 1)
    y_train = train_scaled[1:].reshape(-1, 1)

    # Build and train model
    model = Sequential([
        Input(shape=(1, 1)),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=MeanSquaredError())
    model.fit(X_train, y_train, epochs=100, verbose=0)

    # Sequentially predict test revenues
    # Start from last training revenue scaled value
    input_seq = train_scaled[-1].reshape(1, 1, 1)
    predictions_scaled = []

    for _ in range(len(test_revenues)):
        pred_scaled = model.predict(input_seq)
        predictions_scaled.append(pred_scaled[0, 0])
        # Next input is current prediction
        input_seq = pred_scaled.reshape(1, 1, 1)

    predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions_scaled).flatten()

    # Calculate RMSE between actual and predicted test revenues
    rmse = np.sqrt(mean_squared_error(test_revenues, predictions))

    # Also calculate % RMSE relative to mean test revenue
    percent_rmse = (rmse / np.mean(test_revenues)) * 100

    print(f"{startup_name} test RMSE: {rmse:.3f} ({percent_rmse:.2f}%)")

    # Save model and scaler
    model.save(f"{base_dir}/models/forecast_{startup_name}.h5")
    joblib.dump(scaler, f"{base_dir}/models/scaler_{startup_name}.pkl")
