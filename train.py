import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# -----------------------------
# 1. Đọc dữ liệu
# -----------------------------
DATA_PATH = "data/PJME_hourly.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Không tìm thấy file {DATA_PATH}")

df = pd.read_csv(DATA_PATH, parse_dates=['Datetime'], index_col='Datetime')
df.rename(columns={'PJME_MW': 'Power'}, inplace=True)

# -----------------------------
# 2. Tạo feature thời gian
# -----------------------------
df['hour'] = df.index.hour
df['day'] = df.index.day
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
df['year'] = df.index.year

# Chỉ giữ các feature thời gian, loại bỏ lag
X = df[["hour","day","dayofweek","month","year"]]
y = df["Power"]

# -----------------------------
# 3. Chia dữ liệu train/test
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# -----------------------------
# 4. Chuẩn hóa
# -----------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 5. Huấn luyện Lasso Regression
# -----------------------------

lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mse_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)
print(f"Lasso -> RMSE: {rmse_lasso:.2f}, MAE: {mae_lasso:.2f}, R2: {r2_lasso:.4f}")

# -----------------------------
# 6. Huấn luyện XGBoost
# -----------------------------

xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)

mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
print(f"XGBoost -> RMSE: {rmse_xgb:.2f}, MAE: {mae_xgb:.2f}, R2: {r2_xgb:.4f}")

# -----------------------------
# 7. So sánh
# -----------------------------
print("\n===== So sánh mô hình =====")
print(f"Lasso   -> RMSE: {rmse_lasso:.2f}, MAE: {mae_lasso:.2f}, R2: {r2_lasso:.4f}")
print(f"XGBoost -> RMSE: {rmse_xgb:.2f}, MAE: {mae_xgb:.2f}, R2: {r2_xgb:.4f}")

# -----------------------------
# 8. Vẽ biểu đồ
# -----------------------------
plt.figure(figsize=(14,6))
plt.plot(y_test.values, label="Thực tế", alpha=0.8)
plt.plot(y_pred_lasso, label="Lasso", alpha=0.7)
plt.plot(y_pred_xgb, label="XGBoost", alpha=0.7)
plt.legend()
plt.title("So sánh dự đoán Lasso vs XGBoost")
plt.xlabel("Index")
plt.ylabel("Công suất (MW)")
plt.show()

# -----------------------------
# 9. Lưu model & scaler
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(lasso, "models/lasso_model.pkl")
joblib.dump(xgb_model, "models/xgb_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("✅ Lưu thành công vào thư mục models/")
