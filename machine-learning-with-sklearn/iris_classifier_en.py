import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

### LOADING THE DATA
# Here, the `X` matrix contains the features, while `y` represents the target labels.
# The `as_frame=True` arguments returns the data in a Pandas DataFrame format for us to
# easily work with it.
X, y = load_iris(as_frame=True).data, load_iris(as_frame=True).target

### DIVISION BETWEEN TRAIN AND TEST
# Will divide the dataset between these four variables with 75% of it for train and 25%
# for the test split.
#
# Obs.: Training and testing a model with the same data is a metodological error that leads to
# overfitting with the model failing to generalize to new data.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=18)

### PRE-PROCESSING AND TRANSFORMATION
# Many algorithms work better when all the data is in the same scale. Scikit-learn provides ob-
# jects called transformers with `fit()` and `transform()` methods for this tasks.
#
# It's fundamental for the transformation, like standardization, learn the parameters only in
# the training set and apply this transformation to both the training data and the retained test
# data.
scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

### SELECT THE ESTIMATOR & MODEL TRAINMENT
# The scikit-learn native algorithms are called estimators, some of them are:
# + Random Forest
# + Support Vector Machine
# + K-Nearest Neighbors
#
# Every estimator can be trained and adjusted to the data using `fit(X, y)`.

# Instantiate a KNN model with 5 neighbors.
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model using the standardized data.
knn.fit(X_train, y_train)

### PREDICTION AND ASSESSMENT
# Once the estimator is trained you do not need to retrain it in order to predict
# the results to new data.

# Realize the prediction for the test split.
y_pred = knn.predict(X_test)

# Calculate and print the accuracy score.
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# EXERCISES
# 1. Explain the difference between the StandardScaler and at least two other
#    scalers from Scikit-learn.
# 2. Explain what the `fit()` and `transform()` functions do.
