import joblib
import numpy as np
import os

# Load models once when server starts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

credit_model = joblib.load(os.path.join(BASE_DIR, "models", "credit_model.pkl"))
loan_model = joblib.load(os.path.join(BASE_DIR, "models", "loan_model.pkl"))


def predict_credit_and_loan(data):
    """
    data = {
        landSize,
        income,
        existingLoan,
        repaymentRate,
        previousLoans,
        soilPH,
        rainfall,
        profitability
    }
    """

    features = np.array([[
        data["landSize"],
        data["income"],
        data["existingLoan"],
        data["repaymentRate"],
        data["previousLoans"],
        data["soilPH"],
        data["rainfall"],
        data["profitability"]
    ]])

    credit_score = credit_model.predict(features)[0]
    eligible_loan = loan_model.predict(features)[0]

    return {
        "credit_score": float(round(credit_score, 2)),
        "eligible_loan": float(round(eligible_loan, 2))
    }