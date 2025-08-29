from flask import Flask, render_template, request
import numpy as np
import joblib
import os
import pandas as pd

app = Flask(__name__)

# -----------------------------
# Load model và scaler
# -----------------------------

MODEL_DIR = "models"
xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
lasso_model = joblib.load(os.path.join(MODEL_DIR, "lasso_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_xgb = None
    prediction_lasso = None
    error_msg = None

    if request.method == "POST":
        try:
            # -----------------------------
            # Lấy dữ liệu từ form
            # -----------------------------
            hour = int(request.form["hour"])
            day = int(request.form["day"])
            month = int(request.form["month"])
            year = int(request.form["year"])
            algorithm = request.form["algorithm"]

            # -----------------------------
            # Tạo dataframe đầu vào khớp model (5 feature)
            # -----------------------------
            timestamp = pd.Timestamp(year=year, month=month, day=day)
            input_df = pd.DataFrame([{
                "hour": hour,
                "day": day,
                "dayofweek": timestamp.dayofweek,
                "month": month,
                "year": year,
            }])

            # -----------------------------
            # Chuẩn hóa dữ liệu
            # -----------------------------
            X_scaled = scaler.transform(input_df)

            # -----------------------------
            # Dự đoán
            # -----------------------------
            
            if algorithm == "xgb":
                prediction_xgb = float(xgb_model.predict(X_scaled)[0])
            elif algorithm == "lasso":
                prediction_lasso = float(lasso_model.predict(X_scaled)[0])
            elif algorithm == "both":
                prediction_xgb = float(xgb_model.predict(X_scaled)[0])
                prediction_lasso = float(lasso_model.predict(X_scaled)[0])

        except Exception as e:
            error_msg = str(e)
            print("Lỗi:", e)

    return render_template(
        "index.html",
        prediction_xgb=prediction_xgb,
        prediction_lasso=prediction_lasso,
        error_msg=error_msg,
        form_data=request.form
    )

if __name__ == "__main__":
    app.run(debug=True)
