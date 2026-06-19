"""
NBA Player Data Analysis & Preprocessing

Author: Diogo Bonofre (@ssevverall)

Description:
    This script performs Exploratory Data Analysis (EDA) and preprocessing
    experiments on NBA player statistics sourced from a CSV dataset.

    It serves as an exercise in understanding:
    1. Remote Data Sourcing: Loading data directly from a Google Sheets CSV export.
    2. Exploratory Data Analysis (EDA): Using .head(), .info(), and .describe()
       to understand the dataset's "shape" and health.
    3. Data Visualization: Leveraging Matplotlib and Pandas plotting wrappers
       (histograms, boxplots) to identify distributions and outliers.
    4. Data Cleaning: Identifying missing (null) values and strategies for
       handling statistical anomalies (outliers).
    5. Feature Scaling: Comparing ranges of different attributes to determine
       the necessity of normalization or standardization.

File Format (CSV):
    [Column]        [Type]          [Description]
    Age             Integer         Player's age during the season
    Year            Integer         The specific NBA season
    FG              Float           Field Goals made per game
    FGA             Float           Field Goal Attempts per game
    ...             ...             ...

Reference:
    - Kaggle EDA Guide: https://www.kaggle.com/code/kashnitsky/topic-1-exploratory-data-analysis-with-pandas
    - Pandas Documentation: https://pandas.pydata.org/docs/
"""

import matplotlib.pyplot as plt
import pandas as pd

# We use the export?format=csv parameter to pull a Google Sheet directly into a DataFrame
url = "https://docs.google.com/spreadsheets/d/13rdHPzx5B9BuPcRkncDxM5gyK-2VJKbaPhXyAfYusBM/export?format=csv"
df = pd.read_csv(url)

# Print the first 5 rows to verify the data loaded correctly
print(df.head())

# info() is crucial: it shows the data type of every column and counts non-null values
print(df.info())

# Visualizing distributions
# Histograms help us see if data is "Normal" (bell curve) or skewed (piled to one side)
df[["Age", "Year"]].hist()
plt.suptitle("Distribution of Age and Year")

# Outlier detection
# Boxplots are the gold standard for spotting outliers (the individual dots outside the whiskers)
df[["FG", "FGA"]].boxplot()
plt.title("Outlier Detection for FG and FGA")


# Handling missing values
# Check which columns have "NaN" (Not a Number) entries
print(df.isnull().sum())

# Treating outliers
# One common way to treat outliers is "Clipping" or "Capping" using the IQR (Interquartile Range)
# Formula: IQR = Q3 - Q1. Any value > Q3 + 1.5*IQR is usually considered an outlier.
# Example: df['FG'] = df['FG'].clip(lower=df['FG'].quantile(0.05), upper=df['FG'].quantile(0.95))

# Scaling and distribution
# describe() shows the scale (mean, min, max).
# If 'Age' ranges 19-40 and 'Salary' ranges 0-50,000,000, we need scaling for Machine Learning.
print(df.describe())

# Display all generated plots
plt.show()
