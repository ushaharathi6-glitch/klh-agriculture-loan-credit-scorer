from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ROUTE FOR FRONTEND
@app.route("/")
def home():
    return render_template("index.html")

# PREDICT ROUTE
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    income = data.get("income", 0)
    land = data.get("landSize", 0)

    credit_score = min(100, int((income / 10000) + (land * 2)))
    eligible_loan = int(income * 2)

    return jsonify({
        "credit_score": credit_score,
        "eligible_loan": eligible_loan
    })

if __name__ == "__main__":
    app.run(debug=True)