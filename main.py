import pickle
import warnings
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# 03. Load Dataset
df = pd.read_csv("dataset.csv")

# Standardize column names (lowercase & stripped)
df.columns = df.columns.str.strip().str.lower()

# Map common variations if present
column_map = {
    "applicantincome": "applicant_income",
    "coapplicantincome": "coapplicant_income",
    "loanamount": "loan_amount",
    "loan_amount_term": "loan_term",
}
df.rename(columns=column_map, inplace=True)

# 04. Data Understanding
print("First 5 records:\n", df.head())
print("\nDataset Shape:", df.shape)
print("\nDataset Info:\n")
df.info()
print("\nNumerical Summary:\n", df.describe())
print("\nMissing Values Count:\n", df.isnull().sum())

# Remove Duplicates
df = df.drop_duplicates()

# 08. Remove Irrelevant Columns (Loan_ID)
if "loan_id" in df.columns:
    df = df.drop(columns=["loan_id"])

# 07. Handle Missing Values
cat_cols = [
    "gender",
    "married",
    "dependents",
    "education",
    "self_employed",
    "property_area",
]
num_cols = ["applicant_income", "coapplicant_income", "loan_amount", "loan_term"]

for col in cat_cols:
  if col in df.columns:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in num_cols:
  if col in df.columns:
    df[col] = df[col].fillna(df[col].median())

if "credit_history" in df.columns:
  df["credit_history"] = df["credit_history"].fillna(
      df["credit_history"].mode()[0]
  )

# 05. Visualizations / EDA
plt.figure(figsize=(6, 4))
sns.countplot(x="loan_status", data=df)
plt.title("Target Distribution: Approved vs Rejected")
plt.show()

# 09. Encode Categorical Data
encoders = {}
for col in cat_cols:
  if col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Map Target: Y -> 1 (Approved), N -> 0 (Rejected)
df["loan_status"] = df["loan_status"].astype(str).str.upper().map({"Y": 1, "N": 0})
df = df.dropna(subset=["loan_status"])
df["loan_status"] = df["loan_status"].astype(int)

# 06. Identify X and y
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# 10. Train / Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standard Scaling (Crucial: prevents 50,000 from overflowing the model)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 11 & 12. Build & Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 13. Make Predictions
y_pred = model.predict(X_test_scaled)

# 14 & 15. Evaluation & Metrics
print("\n--- Model Evaluation ---")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-Score : {f1_score(y_test, y_pred, zero_division=0):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Rejected", "Approved"],
    yticklabels=["Rejected", "Approved"],
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 16. Test New Application (From Assignment Specification)
sample_applicant = pd.DataFrame([{
    "gender": encoders["gender"].transform(["Male"])[0],
    "married": encoders["married"].transform(["Yes"])[0],
    "dependents": encoders["dependents"].transform(["0"])[0],
    "education": encoders["education"].transform(["Graduate"])[0],
    "self_employed": encoders["self_employed"].transform(["No"])[0],
    "applicant_income": 50000,
    "coapplicant_income": 0,
    "loan_amount": 200,  # 2 Lakhs in Thousands
    "loan_term": 360,
    "credit_history": 1.0,
    "property_area": encoders["property_area"].transform(["Urban"])[0],
}])

sample_applicant = sample_applicant[list(X.columns)]
sample_applicant_scaled = scaler.transform(sample_applicant)
sample_pred = model.predict(sample_applicant_scaled)[0]
sample_prob = model.predict_proba(sample_applicant_scaled)[0]

print("\n--- Test Sample Prediction ---")
print(
    f"Status: {'Approved (Y)' if sample_pred == 1 else 'Rejected (N)'}"
    f" (Approval Chance: {sample_prob[1]*100:.1f}%)"
)

# Save Complete Pipeline Bundle
bundle = {
    "model": model,
    "scaler": scaler,
    "feature_names": list(X.columns),
    "encoders": encoders,
}

with open("loan_approval_model.pkl", "wb") as f:
  pickle.dump(bundle, f)

print("\nModel pipeline bundled & saved to loan_approval_model.pkl")