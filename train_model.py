import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

# ---------------------------------
# Setup Paths (IMPORTANT FIX)
# ---------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "merged_farmer_credit_loan_training_dataset_10000_rows.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------
# Load Dataset
# ---------------------------------
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("Dataset Loaded ✅")
print("Shape:", df.shape)

# ---------------------------------
# Create Loan Eligibility Target
# ---------------------------------
income_component = df["Annual_Income"] * np.random.uniform(0.30, 0.45)
weather_factor = df["Rainfall_2022_mm"] * np.random.uniform(10, 20)
profit_history = df["Profitability_2022"] * np.random.uniform(0.2, 0.4)
experience_bonus = np.log1p(df["Previous_Loan_Count"] + 1) * 10000
risk_penalty = df["Existing_Loan_Amount"] * np.random.uniform(0.9, 1.2)

base_loan = (
    income_component
    + weather_factor
    + profit_history
    + experience_bonus
    - risk_penalty
)

noise = np.random.normal(0, base_loan.std() * 0.30, len(df))
df["Eligible_Loan_Amount"] = (base_loan + noise).clip(lower=0)

# ---------------------------------
# Remove Leakage
# ---------------------------------
leakage_cols = [
    "Farmer_ID",
    "Synthetic_Credit_Score",
    "Synthetic_Loan_Amount"
]

for col in leakage_cols:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# ---------------------------------
# Feature Selection
# ---------------------------------
features = [
    "Land_Size_Acres",
    "Annual_Income",
    "Existing_Loan_Amount",
    "Repayment_Rate",
    "Previous_Loan_Count",
    "Soil_pH",
    "Rainfall_2023_mm",
    "Profitability_2023"
]

X = df[features]

# =================================
# MODEL 1 — CREDIT SCORE
# =================================
y_credit = df["Credit_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_credit,
    test_size=0.2,
    random_state=42
)

credit_model = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

credit_model.fit(X_train, y_train)

y_pred_credit = credit_model.predict(X_test)

print("\n===== Credit Score Model =====")
print("R2:", r2_score(y_test, y_pred_credit))
print("MAE:", mean_absolute_error(y_test, y_pred_credit))

joblib.dump(credit_model, os.path.join(MODEL_DIR, "credit_model.pkl"))

# =================================
# MODEL 2 — LOAN ELIGIBILITY
# =================================
y_loan = df["Eligible_Loan_Amount"]

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y_loan,
    test_size=0.2,
    random_state=42
)

loan_model = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

loan_model.fit(X_train2, y_train2)

y_pred_loan = loan_model.predict(X_test2)

print("\n===== Loan Amount Model =====")
print("R2:", r2_score(y_test2, y_pred_loan))
print("MAE:", mean_absolute_error(y_test2, y_pred_loan))

joblib.dump(loan_model, os.path.join(MODEL_DIR, "loan_model.pkl"))

print("\nBoth models saved successfully ✅")