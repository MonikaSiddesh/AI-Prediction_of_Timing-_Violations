import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    max_error,
)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import time

# --- 1. Data Loading and Preprocessing ---

DATASET_PATH = "/content/logic_depth_dataset.csv"  

try:
    data = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print(f"Error: File not found at {DATASET_PATH}.  Please check the path.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- 2. Data Validation and Cleaning ---
data.dropna(inplace=True)
data.drop('Circuit_ID', axis=1, inplace=True)

# --- 3. Feature Engineering: Optimized One-Hot Encoding ---
gate_types = data['Gate_Types'].str.get_dummies(sep=',')
gate_types.columns = [col.replace(' ', '_') for col in gate_types.columns]
data = pd.concat([data, gate_types], axis=1)
data.drop('Gate_Types', axis=1, inplace=True)

# --- 4. Feature Selection ---
gate_type_features = list(gate_types.columns)
features = ['Num_Gates', 'Max_Path_Length'] + gate_type_features
X = data[features]
y = data['Logic_Depth']

# --- 5. Data Scaling ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 6. Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# --- 7. Model Training (XGBoost) with Early Stopping (on final model) ---

#  Validation set for early stopping *after* the main train/test split.
X_train_es, X_val_es, y_train_es, y_val_es = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)

xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
)

start_time = time.time()
xgb_model.fit(
    X_train_es,
    y_train_es,
    eval_set=[(X_val_es, y_val_es)],  # Validation set for early stopping
    verbose=False,
)
end_time = time.time()
training_time = end_time - start_time
print(f"Training Time: {training_time:.2f} seconds")


# --- 8. Model Evaluation (Multiple Metrics) ---

y_pred = xgb_model.predict(X_test)
metrics = {
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    "MAE": mean_absolute_error(y_test, y_pred),
    "R-squared": r2_score(y_test, y_pred),
    "Explained Variance": explained_variance_score(y_test, y_pred),
    "Max Error": max_error(y_test, y_pred),
}

print("\nEvaluation Metrics:")
for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")

# --- 9. Cross-Validation (WITHOUT Early Stopping) ---

#  Remove early_stopping_rounds from the XGBRegressor *during* cross-validation.
xgb_model_cv = xgb.XGBRegressor(  # Create a *separate* instance for CV
    objective='reg:squarederror',
    n_estimators=100,  #  A reasonable number of estimators for CV
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    # NO early_stopping_rounds here!
)

scoring_metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
cv_results = cross_validate(
    xgb_model_cv, X_scaled, y, cv=5, scoring=scoring_metrics, n_jobs=-1
)

print("\nCross-Validation Results:")
print(f"  CV RMSE: {np.sqrt(-cv_results['test_neg_mean_squared_error'].mean()):.4f}")
print(f"  CV MAE: {-cv_results['test_neg_mean_absolute_error'].mean():.4f}")
print(f"  CV R-squared: {cv_results['test_r2'].mean():.4f}")

# --- 10. Visualization (Same as before) ---
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Logic Depth")
plt.ylabel("Predicted Logic Depth")
plt.title("Predicted vs. Actual")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_test, residuals, alpha=0.5)
plt.xlabel("Actual Logic Depth")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.axhline(y=0, color='k', linestyle='--', lw=2)
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x=xgb_model.feature_importances_, y=features)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance")
plt.show()

# --- 11. Prediction on New Data (Same as before) ---
new_data_dict = {
    'Num_Gates': [30],
    'Max_Path_Length': [15],
    'NAND': [10],
    'XNOR': [5],
    'NOT': [5],
    'AND': [5],
    'OR': [3],
    'XOR': [2],
    'NOR': [0]
}

for col in gate_type_features:
    if col not in new_data_dict:
        new_data_dict[col] = [0]

new_data = pd.DataFrame(new_data_dict)
new_data = new_data[features]
new_data_scaled = scaler.transform(new_data)
predicted_depth = xgb_model.predict(new_data_scaled)
print(f"\nPredicted Logic Depth for New Data: {predicted_depth[0]:.2f}")