## Author: Diogo G. Bonofre dos Santos (@hayes)
## Created: 2026-06-09

### Commentary:
## This script explores the fundamental differences between class distributions
## using NumPy for vectorized statistical calculations and Matplotlib for visual
## overlay. It serves as a bridge between raw data metrics and the mathematical
## intuition required for classification algorithms.
##
## Statistical Insights:
##
## The centers of the distributions are significantly separated (median of 12.2
## for benign versus 17.3 for malignant), indicating that 'mean radius' is a
## highly discriminative feature. If the histograms overlapped entirely, the
## feature would yield no information gain for classification.
##
## However, the intersecting region (roughly between 13.5 and 15.5) represents a
## state of higher entropy. In this zone, a simple 1D threshold guarantees
## classification errors (False Positives and False Negatives) due to local
## geometric ambiguity.
##
## This overlap demonstrates why models rarely rely on a single variable. While
## two tumors might be indistinguishable at a 14.5 radius on a 1D plane,
## introducing a second orthogonal feature (like mean texture) allows the points
## to separate vertically. This geometric separation is the exact mechanism that
## allows the K-Nearest Neighbors (KNN) algorithm to succeed where a simple 1D
## threshold fails.

### Code:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# Now, everytime we run the code, the window will spawn and update dinamically
plt.ion()
# NOTE: A useful workflow in Emacs is to spawn a Python REPL buffer with
# `run-python` and them submit code blocks with `python-shell-send-buffer`

# Load data and convert to DataFrame
data = load_breast_cancer(as_frame=True)
df = data.frame

# Separate the classes them convert them to NumPy arrays for mathematical
# operations
radius_benign = df[df["target"] == 1]["mean radius"].to_numpy()
radius_malignant = df[df["target"] == 0]["mean radius"].to_numpy()

# Applying these operations directly on NumPy arrays is computationally faster
# than relying on Pandas high-level methods like `.describe()`
rb_mean = np.mean(radius_benign)
rb_median = np.median(radius_benign)
print(f"Radius Benign (Mean: {rb_mean:.4}, Median: {rb_median:.4})")

rm_mean = np.mean(radius_malignant)
rm_median = np.median(radius_malignant)
print(f"Radius Malignant (Mean: {rm_mean:.4}, Median: {rm_median:.4})")

# Initialize Matplotlib canvas
fig, ax = plt.subplots(figsize=(8, 6))

# Overlay method using sequential calls and `alpha` parameter
ax.hist(radius_benign, bins=20, alpha=0.6, label="Benign", color="tab:blue")
ax.hist(
    radius_malignant, bins=20, alpha=0.6, label="Malignant", color="tab:red"
)

# Adding essential context to the canvas
ax.set_title("Distribution of Mean Radius: Benign vs Malignant")
ax.set_xlabel("Mean Radius")
ax.set_ylabel("Frequency")
ax.legend()
