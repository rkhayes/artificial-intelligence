import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer

# EXPLORATORY DATA ANALYSIS
# Loading the data
X, y = load_breast_cancer(as_frame=True).data, load_breast_cancer(as_frame=True).target

# Let's look all the data contained in the dataset:
print(X.head())
# The `head()` methods shows us only the 5 first rows of the `DataFrame`.

# Now, let's assume that the most important metrics are `mean radius` and `mean texture`
# and filter our data table to look only at these two columns.
X = X.iloc[:, :2]
print(X)
# We're doing this to reduce dimensionality in our problem. Working with 2 dimensions is
# easier because we can represent data in a 2D euclidian plane.
#
# To understand how the `iloc` property works, read:
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html

# We will now concatenate the data (X/features) with the labels (targets/y) to see better
# how is the relation between them.
data = pd.concat([X, y], axis=1)
print(data)

# Here we are mapping all the 0's and 1's from the our labels to more descriptive values.
data["target"] = data["target"].map({1: "Benign", 0: "Malignant"})
print(data)

# Instead of always looking at the top and bottom values of our DataFrame, we can use a
# random sample of itens. Actually, this is exactly what the `sample` method is for.
data.sample(n=5, random_state=18)
# Here we use `n` to define the number of samples to return and `random_state` for repro-
# ducibility in the sample visualized.

# Ok, it's enough of lines and columns. Let's plot the data!

# PLOTTING THE DATA
sns.scatterplot(data, x="mean radius", y="mean texture", hue="target")
sns.pairplot(data, hue="target")
data.hist()
plt.show()
