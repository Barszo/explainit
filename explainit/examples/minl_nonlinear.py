import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load the California Housing dataset
print("Loading California Housing dataset...")
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Create DataFrame for better understanding
feature_names = housing.feature_names
df = pd.DataFrame(X, columns=feature_names)
df['PRICE'] = y

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataset description:")
print(df.describe())
print("\nFeature descriptions:")
print("MedInc: Median income in block group")
print("HouseAge: Median house age in block group")
print("AveRooms: Average number of rooms per household")
print("AveBedrms: Average number of bedrooms per household")
print("Population: Block group population")
print("AveOccup: Average number of household members")
print("Latitude: Block group latitude")
print("Longitude: Block group longitude")

# Prepare the data
print("\nPreparing data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set shape: {X_train_scaled.shape}")
print(f"Test set shape: {X_test_scaled.shape}")

# Build the neural network model
print("\nBuilding the model...")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer for regression (no activation)
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mean_absolute_error']
)

print("Model summary:")
model.summary()

# Train the model
print("\nTraining the model...")
history = model.fit(
    X_train_scaled, y_train,
    batch_size=64,
    epochs=100,
    validation_split=0.2,
    verbose=1
)

# Evaluate the model
print("\nEvaluating the model...")
test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Loss (MSE): {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# Make predictions
print("\nMaking predictions...")
predictions = model.predict(X_test_scaled)

# Calculate additional metrics
from sklearn.metrics import r2_score, mean_squared_error
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mean_absolute_error'], label='Training MAE')
plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()

# Plot predictions vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Prices (hundreds of thousands of dollars)')
plt.ylabel('Predicted Prices (hundreds of thousands of dollars)')
plt.title('Actual vs Predicted House Prices - California Housing')
plt.show()

# Example of using the model for new predictions
print("\nExample prediction for a new house:")
# Take the first test sample as an example
sample_house = X_test_scaled[0:1]
predicted_price = model.predict(sample_house)
actual_price = y_test[0]

print(f"Predicted price: ${predicted_price[0][0]:.2f} (hundreds of thousands)")
print(f"Actual price: ${actual_price:.2f} (hundreds of thousands)")
print(f"Difference: ${abs(predicted_price[0][0] - actual_price):.2f} (hundreds of thousands)")

# Feature importance visualization
print("\nFeature names and their indices:")
for i, feature in enumerate(feature_names):
    print(f"{i}: {feature}")

# Show sample predictions with feature values
print("\nSample house features and prediction:")
sample_idx = 0
print("Features:")
for i, (feature, value) in enumerate(zip(feature_names, X_test[sample_idx])):
    print(f"  {feature}: {value:.4f}")
print(f"Predicted price: ${predictions[sample_idx][0]:.2f} hundred thousands")
print(f"Actual price: ${y_test[sample_idx]:.2f} hundred thousands")


# Calculate various efficiency metrics
def calculate_regression_efficiency(y_true, y_pred):
    """Calculate comprehensive efficiency metrics for regression"""
    
    # Basic metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Additional metrics
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error
    
    # Efficiency relative to baseline (mean prediction)
    baseline_mse = mean_squared_error(y_true, [np.mean(y_true)] * len(y_true))
    efficiency_ratio = 1 - (mse / baseline_mse)  # How much better than baseline
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape,
        'Efficiency vs Baseline': efficiency_ratio
    }

# Use with your model
metrics = calculate_regression_efficiency(y_test, predictions.flatten())
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")