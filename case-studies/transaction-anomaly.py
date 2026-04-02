# Case Study 4: Transaction Anomaly Detection
# CIBC's monitoring team wants to detect unusual transaction patterns without labeled fraud data. You're given 6 
# months of transaction data with: amount, hour_of_day, merchant_category, location, frequency_last_7d, 
# avg_txn_amount_30d, distance_from_home_km. There are no labels — this is unsupervised. Build an anomaly detection 
# system and explain how you'd set the threshold for alerting.

# import the libraries
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

# load data
df = pd.read_csv("fraud_data.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isna().sum())

# feature engg
# amount deviation
df["amount_ratio"] = df["amount"] / df["customer_avg_txn"] # risky if high
# time deviation
df["is_night"] = df["hour_of_day"].apply(lambda h: 1 if h<6 or h>22 else 0)
# international and high value
df["international"] = (df["is_international"]==1) & (df["amount"] > 5000)
# gap
df["gap"] = (df["days_since_last_txn"] > 30).astype(int)
# high value
df["high_value"] = df["amount"] > 5000

# preprocess
X = df.drop(columns=["transaction_id"])

# train model
isf = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)

# pipeline
pipeline = Pipeline([
    ("model", isf)
])
isf.fit(X)
scores = pipeline.decision_function(X)
df["anomaly_score"] = scores
threshold = df["anomaly_score"].quantile(0.01)
df["alert"] = df["anomaly_score"] < threshold
