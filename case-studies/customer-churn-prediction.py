# Case Study 2: Customer Churn Prediction
# CIBC's retail banking team wants to predict which customers will close their accounts in the next 90 days. 
# You're given a dataset with: account_age_months, num_products, avg_monthly_balance, num_transactions_last_90d, 
# num_support_calls, has_mortgage, has_credit_card, salary_estimate, age, province, and label churned. The churn 
# rate is 8%. Build a model and identify the top features driving churn.

# import the libraries
from sklearn.model_selection import train_test_split
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline

# load data
df = pd.read_csv("customer_churn.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isna().sum())

# feature engg
# recently opened
df["recent"] = (df["account_age_months"] < 12).astype(int)
# concerns
df["frequent_support_calls"] = (df["num_support_calls"] > 3).astype(int)
# has any products
df["has_products"] (df["num_products"]>0).astype(int)

# preprocess
X = df.drop(columns=["churned"]).values
y = df.loc[:, "churned"].values

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X_train, y_train)

# train model
# pipeline
pipeline = Pipeline([
    ("classifier", LogisticRegression())
])
pipeline.fit(X_sm, y_sm)

# predict
y_pred = pipeline.predict(X_test)

# evaluate
print(classification_report(y_test, y_pred))
print(f1_score(y_test, y_pred))