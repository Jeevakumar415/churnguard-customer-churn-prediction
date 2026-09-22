import pandas as pd
from sklearn.linear_model import LogisticRegression


print("=" * 70)
print("ChurnGuard - Task 4: Customer Churn Prediction")
print("=" * 70)


# 1. LOAD RAW DATASET

file_path = "data/churnguard_data.csv"

df = pd.read_csv(file_path)

print("\nRaw dataset loaded successfully.")
print("Original shape:", df.shape)



# 2. CLEAN DATA
# Drop customer ID
df = df.drop(columns=["customerID"])


# Remove duplicate rows
df = df.drop_duplicates()


# Strip whitespace
df["gender"] = df["gender"].str.strip()
df["PaymentMethod"] = df["PaymentMethod"].str.strip()


# Standardise casing
df["Churn"] = df["Churn"].str.strip().str.title()
df["PhoneService"] = df["PhoneService"].str.strip().str.title()
df["PaperlessBilling"] = df["PaperlessBilling"].str.strip().str.title()


# Standardise Contract values
contract_mapping = {
    "Month-to-month": "Month-to-month",
    "Month to month": "Month-to-month",
    "One year": "One year",
    "One Year": "One year",
    "Two year": "Two year",
    "Two Year": "Two year"
}

df["Contract"] = df["Contract"].replace(contract_mapping)


# Standardise InternetService values
internet_mapping = {
    "DSL": "DSL",
    "dsl": "DSL",
    "Fiber optic": "Fiber optic",
    "Fiber Optic": "Fiber optic",
    "No": "No",
    "no": "No"
}

df["InternetService"] = df["InternetService"].replace(
    internet_mapping
)


# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)


# Remove invalid tenure
df = df[df["tenure"] > 0]


# Remove invalid MonthlyCharges
df = df[
    df["MonthlyCharges"].between(10, 200)
]


# Fill missing MonthlyCharges
df["MonthlyCharges"] = df["MonthlyCharges"].fillna(
    df["MonthlyCharges"].mean()
)


# Fill missing TotalCharges
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].mean()
)


# Fill missing tenure
df["tenure"] = df["tenure"].fillna(
    round(df["tenure"].median())
)


print("Cleaned dataset shape:", df.shape)


# 3. PREPARE TARGET

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})


# 4. CONVERT CONTRACT TO REQUIRED 0/1/2 VALUES

contract_numeric_mapping = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

df["Contract"] = df["Contract"].map(
    contract_numeric_mapping
)



# 5. SELECT FIVE REQUIRED FEATURES

features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
    "Contract"
]

X = df[features]
y = df["Churn"]



# 6. TRAIN LOGISTIC REGRESSION MODEL

model = LogisticRegression(max_iter=1000)

model.fit(X, y)

print("\nPrediction model trained successfully.")


# 7. GET CUSTOMER INPUT

print("\n" + "=" * 70)
print("ENTER CUSTOMER DETAILS")
print("=" * 70)

try:

    tenure = float(
        input("\n1. Tenure (months): ")
    )

    monthly_charges = float(
        input("2. Monthly Charges: ")
    )

    total_charges = float(
        input("3. Total Charges: ")
    )

    senior_citizen = int(
        input("4. Senior Citizen (0 = No, 1 = Yes): ")
    )

    contract = int(
        input(
            "5. Contract "
            "(0 = Month-to-month, "
            "1 = One year, "
            "2 = Two year): "
        )
    )


    # ========================================================
    # 8. VALIDATE INPUT
    # ========================================================

    if tenure < 0:
        raise ValueError(
            "Tenure cannot be negative."
        )

    if monthly_charges < 0:
        raise ValueError(
            "Monthly Charges cannot be negative."
        )

    if total_charges < 0:
        raise ValueError(
            "Total Charges cannot be negative."
        )

    if senior_citizen not in [0, 1]:
        raise ValueError(
            "Senior Citizen must be 0 or 1."
        )

    if contract not in [0, 1, 2]:
        raise ValueError(
            "Contract must be 0, 1, or 2."
        )


    # ========================================================
    # 9. CREATE CUSTOMER DATA
    # ========================================================

    customer_data = pd.DataFrame(
        [[
            tenure,
            monthly_charges,
            total_charges,
            senior_citizen,
            contract
        ]],
        columns=features
    )


    # 10. PREDICT CHURN

    prediction = model.predict(
        customer_data
    )[0]

    churn_probability = model.predict_proba(
        customer_data
    )[0][1]


    # 11. DISPLAY RESULT

    print("\n" + "=" * 70)
    print("PREDICTION RESULT")
    print("=" * 70)

    if prediction == 1:

        print("\nLikely to CHURN")

    else:

        print("\nLikely to STAY")


    print(
        f"Churn Probability: {churn_probability:.2%}"
    )

    print("\n" + "=" * 70)
    print("Task 4 completed successfully.")
    print("=" * 70)


except ValueError as e:

    print("\nInput Error:", e)

except Exception as e:

    print("\nUnexpected Error:", e)
