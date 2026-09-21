import pandas as pd


# ChurnGuard - Task 2: Clean the Dataset
print("=" * 70)
print("ChurnGuard - Task 2: Clean the Dataset")
print("=" * 70)


# 1. Load the raw dataset
df = pd.read_csv("data/churnguard_data.csv")

print("\nOriginal dataset shape:")
print(df.shape)


# Create a copy for cleaning
clean = df.copy()


# 2. Drop customerID
clean = clean.drop(columns=["customerID"])


# 3. Remove duplicate rows
clean = clean.drop_duplicates()


# 4. Strip whitespace from selected columns
for col in ["gender", "PaymentMethod"]:
    if col in clean.columns:
        clean[col] = clean[col].astype(str).str.strip()

# 5. Standardize casing
for col in ["Churn", "PhoneService", "PaperlessBilling"]:
    if col in clean.columns:
        clean[col] = (
            clean[col]
            .astype(str)
            .str.strip()
            .str.title()
        )


# 6. Standardize Contract values
if "Contract" in clean.columns:

    contract_map = {
        "Month to month": "Month-to-month",
        "Month-to-month": "Month-to-month",
        "One year": "One year",
        "Two year": "Two year"
    }

    clean["Contract"] = (
        clean["Contract"]
        .astype(str)
        .str.strip()
        .replace(contract_map)
    )


# 7. Standardize InternetService values
if "InternetService" in clean.columns:

    internet_map = {
        "DSL": "DSL",
        "dsl": "DSL",
        "Fiber optic": "Fiber optic",
        "fiber optic": "Fiber optic",
        "Fiber Optic": "Fiber optic",
        "No": "No",
        "no": "No"
    }

    clean["InternetService"] = (
        clean["InternetService"]
        .astype(str)
        .str.strip()
        .replace(internet_map)
    )


# 8. Convert TotalCharges to numeric
clean["TotalCharges"] = pd.to_numeric(
    clean["TotalCharges"],
    errors="coerce"
)


# 9. Remove invalid tenure values
clean = clean[clean["tenure"] > 0]


# 10. Remove invalid MonthlyCharges values
clean = clean[
    clean["MonthlyCharges"].between(10, 200)
]

# 11. Fill missing MonthlyCharges with mean
clean["MonthlyCharges"] = clean["MonthlyCharges"].fillna(
    clean["MonthlyCharges"].mean()
)

# 12. Fill missing TotalCharges with mean
clean["TotalCharges"] = clean["TotalCharges"].fillna(
    clean["TotalCharges"].mean()
)


# 13. Fill missing tenure with rounded median
clean["tenure"] = clean["tenure"].fillna(
    round(clean["tenure"].median())
).astype(int)


# Print final results
print("\n" + "=" * 70)
print("CLEANING COMPLETED")
print("=" * 70)

print("\nFinal dataset shape:")
print(clean.shape)

print("\nMissing values after cleaning:")
print(clean.isna().sum())


# Display final column information
print("\nFinal data types:")
clean.info()


# Display cleaned Contract values
print("\nContract values after cleaning:")
print(clean["Contract"].unique())


# Display cleaned Churn values
print("\nChurn values after cleaning:")
print(clean["Churn"].value_counts(dropna=False))


print("\n" + "=" * 70)
print("Task 2 completed successfully.")
print("=" * 70)