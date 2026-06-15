## Author: Diogo Bonofre (@hayes)
## Created: 26-06-15
## 
## Commentary:
## This script demonstrates a manual "forward pass" of a single-node
## model. It exposes how predictions are generated via linear
## combinations (y = wx + b) and how the error is quantified using
## Mean Squared Error (MSE).
##
## Code:
import numpy as np

def predict(x, w, b):
    """Generates a prediction using linear equation ŷ = wx + b."""
    return (w * x) + b

# Translation of the mathematical formula L = 1/n * Σ(y - ŷ)²
def calculate_mse(y_true, y_pred):
    """Calculates the Mean Squared Error between true and predicted values."""
    # We use np.mean to handle the (1/n * Σ) portion of the formula automatically
    squared_errors = (y_true - y_pred) ** 2
    return np.mean(squared_errors)

# Dummy data (e.g., x = tumor size, y = severity score)
X = np.array([1.2, 2.4, 3.1, 4.5, 5.0])
y_true = np.array([2.0, 4.5, 5.9, 8.8, 10.1])

# Initialize parameters with random guesses
weight = 0.5
bias = 1.0

# Perform a "forward pass" (make predictions based on current parameters)
y_pred = predict(X, weight, bias)

# Calculate how wrong our random guesses were
loss = calculate_mse(y_true, y_pred)

print(f"Initial Weight: {weight}")
print(f"Initial Bias: {bias}")
print(f"Predictions: {y_pred}")
print(f"Current Model Loss (MSE): {loss:.4f}")
