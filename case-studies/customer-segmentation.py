# Case Study 5: Customer Segmentation for Marketing
# CIBC's marketing team wants to segment customers for targeted campaigns. Dataset fields: age, income_bracket, 
# num_products, avg_monthly_balance, num_transactions, channel_preference (branch/online/mobile), province, 
# account_type (chequing/savings/both). No labels — this is clustering. Build a segmentation model, determine 
# the optimal number of segments, and describe each segment's profile.

# import the libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# load data
df = pd.read_csv("customer_segmentation.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isna().sum())

# preprocess
ohe = OneHotEncoder()
categorical = ["income_bracket","channel_preference","province","account_type"]
numeric = ["age","num_products","avg_monthly_balance","num_transactions"]
X_cat = ohe.fit_transform(df[categorical]).toarray()

sc = StandardScaler()
num_scaled = sc.fit_transform(df[numeric])
X = np.hstack([num_scaled, X_cat])

scores = []
for k in range(2,10):
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(X)
    scores.append(silhouette_score(X, labels))

# train model
# pipeline
pipeline = Pipeline([
    ("model", KMeans(n_clusters=np.argmax(scores)+2))
])
y_kmeans = pipeline.fit_predict(X)

df["segment"] = y_kmeans
segment_profile = df.groupby("segment").mean()
print(segment_profile)

# evaluate
print(silhouette_score(X, y_kmeans))