import pandas as pd


# ChurnGuard - Task 1: Load and Explore the Dataset

print("=" * 70)
print("ChurnGuard - Task 1: Load and Explore the Dataset")
print("=" * 70)


# 1. Load dataset
df = pd.read_csv("data/churnguard_data.csv")

# 2. Dataset shape
print("\nDataset Shape:")
print(df.shape)


# 3. First 5 rows
print("\nFirst 5 Rows:")
print(df.head())


# 4. Column names and data types
print("\nColumn Names and Data Types:")
df.info()


# 5. Missing-value count for each column
print("\nMissing Values in Each Column:")
print(df.isna().sum())


# 6. Number of duplicate rows
print("\nNumber of Duplicate Rows:")
print(df.duplicated().sum())


# 7. Value counts for Churn
print("\nChurn Value Counts:")
print(df["Churn"].value_counts(dropna=False))


# 8. Unique values in Contract
print("\nUnique Values in Contract:")
print(df["Contract"].unique())