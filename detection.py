# fraud_detection_full_project.py

#  Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

#  Load the dataset
df = pd.read_csv('creditcard.csv')
print(" Data loaded!")

#  Inspect data
print(df.head())
print(df['Class'].value_counts())

#  Check for missing values
print("\nMissing values:\n", df.isnull().sum())

#  Deal with class imbalance using SMOTE
X = df.drop('Class', axis=1)
y = df['Class']

scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])
X['Time'] = scaler.fit_transform(X[['Time']])

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

print("\nResampled dataset shape:", y_res.value_counts())

#  Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

#  Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

#  Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n Model: {name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#  Select best model (Random Forest here) and save it
best_model = models["Random Forest"]
joblib.dump(best_model, 'fraud_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\n Best model saved as 'fraud_model.pkl'")

#  Predict new transaction
def predict_new_transaction(transaction_dict):
    model = joblib.load('fraud_model.pkl')
    scaler = joblib.load('scaler.pkl')

    df_input = pd.DataFrame([transaction_dict])
    df_input['Amount'] = scaler.transform(df_input[['Amount']])
    df_input['Time'] = scaler.transform(df_input[['Time']])

    prediction = model.predict(df_input)

    print("\n Prediction Result:")
    if prediction[0] == 1:
        print(" FRAUDULENT TRANSACTION")
    else:
        print(" LEGITIMATE TRANSACTION")

#  Example input (replace with actual values or use a sample)
sample_transaction = {
    'Time': 50000,
    'V1': -1.359807,
    'V2': -0.072781,
    'V3': 2.536346,
    'V4': 1.378155,
    'V5': -0.338321,
    'V6': 0.462388,
    'V7': 0.239599,
    'V8': 0.098698,
    'V9': 0.363787,
    'V10': 0.090794,
    'V11': -0.5516,
    'V12': -0.617801,
    'V13': -0.99139,
    'V14': -0.311169,
    'V15': 1.468177,
    'V16': -0.470401,
    'V17': 0.207971,
    'V18': 0.025791,
    'V19': 0.403993,
    'V20': 0.251412,
    'V21': -0.018307,
    'V22': 0.277838,
    'V23': -0.110474,
    'V24': 0.066928,
    'V25': 0.128539,
    'V26': -0.189115,
    'V27': 0.133558,
    'V28': -0.021053,
    'Amount': 149.62
}

predict_new_transaction(sample_transaction)

# Visualize class distribution before and after SMOTE
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(x='Class', data=df, ax=axs[0])
axs[0].set_title('Before Resampling')
sns.countplot(x=y_res, ax=axs[1])
axs[1].set_title('After Resampling with SMOTE')
plt.show()
