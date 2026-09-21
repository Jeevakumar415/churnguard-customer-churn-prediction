import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ChurnGuard - Task 3: Train Logistic Regression
print("=" * 70)
print("ChurnGuard - Task 3: Train Logistic Regression")
print("=" * 70)


# 1. Load the raw dataset
df = pd.read_csv("data/churnguard_data.csv")


# 2. Clean the dataset
clean = df.copy()

# Drop customerID
clean = clean.drop(columns=["customerID"])

# Remove duplicate rows
clean = clean.drop_duplicates()

# Strip whitespace
for col in ["gender", "PaymentMethod"]:
    if col in clean.columns:
        clean[col] = clean[col].astype(str).str.strip()

# Standardize casing
for col in ["Churn", "PhoneService", "PaperlessBilling"]:
    if col in clean.columns:
        clean[col] = (
            clean[col]
            .astype(str)
            .str.strip()
            .str.title()
        )

# Standardize Contract
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

# Standardize InternetService
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

# Convert TotalCharges to numeric
clean["TotalCharges"] = pd.to_numeric(
    clean["TotalCharges"],
    errors="coerce"
)

# Remove invalid tenure
clean = clean[clean["tenure"] > 0]

# Remove invalid MonthlyCharges
clean = clean[
    clean["MonthlyCharges"].between(10, 200)
]

# Fill missing MonthlyCharges
clean["MonthlyCharges"] = clean["MonthlyCharges"].fillna(
    clean["MonthlyCharges"].mean()
)

# Fill missing TotalCharges
clean["TotalCharges"] = clean["TotalCharges"].fillna(
    clean["TotalCharges"].mean()
)

# Fill missing tenure
clean["tenure"] = clean["tenure"].fillna(
    round(clean["tenure"].median())
).astype(int)


# ------------------------------------------------------------
# 3. Encode target variable
# ------------------------------------------------------------

clean["Churn"] = clean["Churn"].map({
    "Yes": 1,
    "No": 0
})


# 4. Define required categorical variables

required_categorical_cols = [
    "gender",
    "PhoneService",
    "InternetService",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]


# 5. Include remaining categorical columns
# Logistic Regression requires numerical input.
# The dataset contains additional categorical columns,
# so all categorical feature columns must be encoded.

categorical_cols = clean.select_dtypes(
    include=["object"]
).columns.tolist()


# Make sure the six instructor-specified categorical columns
# are included.
for col in required_categorical_cols:
    if col in clean.columns and col not in categorical_cols:
        categorical_cols.append(col)


# 6. One-hot encode categorical variables
model_df = pd.get_dummies(
    clean,
    columns=categorical_cols,
    drop_first=True
)


# 7. Make sure all feature columns are numeric
X = model_df.drop(columns=["Churn"])
y = model_df["Churn"]

X = X.astype(float)


# 8. Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nDataset shape:", model_df.shape)
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Number of features:", X.shape[1])


# 9. Train Logistic Regression
model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, y_train)


# 10. Make predictions
pred = model.predict(X_test)


# 11. Evaluate model
accuracy = accuracy_score(
    y_test,
    pred
)


print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

print("\nTest Accuracy:")
print(accuracy)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        pred,
        target_names=["Stay", "Churn"],
        zero_division=0
    )
)


print("=" * 70)
print("Task 3 completed successfully.")
print("=" * 70)