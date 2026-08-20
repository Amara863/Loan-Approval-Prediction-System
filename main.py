import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix


import pickle
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv("dataset.csv")
# print(df)
# understand data
# print(df.head())
# print(df.tail())
# print(df.shape)
# print(df.columns)
# print(df.info())
# print(df.describe())
# check missing values
# print(df.isnull().sum())
# print(df.duplicated().sum())
# df=df.drop_duplicates()
# print(df)

# EDA
plt.figure(figsize=(6,4))

sns.countplot(x='Loan_Status', data=df)
plt.title('Target Distribution: Approved vs Rejected')
plt.show()

# Task 7: Handle Missing Values
# Numerical columns mein null values ko mean/median se fill karenge
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
# Categorical columns ko mode se fill karenge
for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term']:
    df[col] = df[col].fillna(df[col].mode()[0])

# Task 8: Remove Unnecessary Columns
# Generally, Loan_ID is an identifier that needs to be removed
if 'Loan_ID' in df.columns:
    df = df.drop('Loan_ID', axis=1)
# Label encoding for target variable (Loan_Status)
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Get Dummies / One-Hot Encoding baki categorical variables ke liye
df = pd.get_dummies(df, drop_first=True)

# define dependent and in dependent variables
x=df.drop('Loan_Status', axis=1)
y=df['Loan_Status']

# split dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,
                                               random_state=42)
# check train ad test data frame shape
print("X_Train: ",x_train.shape)
print("X_test:",x_test.shape)
print("y_train:",y_train.shape)
print("y_test:",y_test.shape)

# create model
model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

# make predictions
y_pred=model.predict(x_test)
print(y_pred)

# Task 14: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# test new data
new_application = pd.DataFrame({
    'ApplicantIncome': [50000],
    'CoapplicantIncome': [0],
    'LoanAmount': [200000],
    'Loan_Amount_Term': [360], # Standard term in most datasets
    'Credit_History': [1],
    'Gender_Male': [1],
    'Married_Yes': [1],
    'Dependents_1': [0],
    'Dependents_2': [0],
    'Dependents_3+': [0],
    'Education_Not Graduate': [0],
    'Self_Employed_Yes': [0],
    'Property_Area_Semiurban': [0],
    'Property_Area_Urban': [1]
})

# Note: The exact columns of 'new_application' must match the columns of 'X' after get_dummies.
# Ensure all columns match by reindexing
new_application = new_application.reindex(columns=x.columns, fill_value=0)

new_prediction = model.predict(new_application)
result = "Approved" if new_prediction[0] == 1 else "Rejected"
print(f"\n--- Prediction for New Application ---")
print(f"The loan application is: {result}")

# save model
with open("loan_approved_status.pkl","wb") as file:
    pickle.dump(model,file)

print("************Model saved to pickle file ************")