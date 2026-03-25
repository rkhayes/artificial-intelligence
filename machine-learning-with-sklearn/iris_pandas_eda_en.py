import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the data in a Pandas DataFrame
iris = load_iris(as_frame=True)

# Complete DataFrame (features + target)
df = iris.frame

# Mapping target to real names of the species for easier reading.
df['species'] = df['target'].map(dict(enumerate(iris.target_names)))

# Checking the features and labels table
print(3 * "5 FIRST LINES")
print(df.head())

# Information about the format of the data
print(3 * "STRUCTURAL INFO")
print(df.info())

# Helps on the numerical detection of outliers
print(3 * "DESCRIPTIVE STATS")
print(df.describe())

# Pairplot: for visualization of the intersected relation between all the features
# and separability of the classes.
sns.pairplot(data=df.drop("target", axis=1), hue="species", diag_kind="kde")
plt.suptitle("Feature Relation (Pairplot)", y=1.02)

# Boxplot: the best to detect visual outliers in each feature.
plt.figure(figsize=(10, 6))
sns.boxplot(data=df.drop(["target", "species"], axis=1), orient="h", palette="Set2")
plt.title("Distribution and detection of outliers")

plt.show()
