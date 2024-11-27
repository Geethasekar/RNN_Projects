# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Generate synthetic data (Sine wave)
def generate_sine_wave_data(timesteps):
    x = np.linspace(0, 50, timesteps)
    y = np.sin(x)
    return y

# Create dataset
timesteps = 500
data = generate_sine_wave_data(timesteps)
plt.plot(data, label='Sine Wave')
plt.title("Synthetic Time Series Data (Sine Wave)")
plt.legend()
plt.show()

# Preprocess data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.reshape(-1, 1))

# Prepare data for LSTM
def create_dataset(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back])
    return np.array(X), np.array(y)

look_back = 20
X, y = create_dataset(data_scaled, look_back)

# Split into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape data for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# Plot training loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Training Loss")
plt.legend()
plt.show()

# Predict
y_pred = model.predict(X_test)

# Inverse transform predictions
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

# Plot predictions vs actual
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted')
plt.title("Predictions vs Actual Values")
plt.legend()
plt.show()

# Save the model
model.save("lstm_time_series_model.h5")
print("Model saved as lstm_time_series_model.h5")
