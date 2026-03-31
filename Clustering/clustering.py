# Use-case: cluster customer support tickets into natural groups before labels 
# exist, or to discover duplicate/overlapping intents. 
# Since BANKING77 is banking-domain text with 77 fine-grained intents, 
# it is ideal for “customer issue grouping” or “intent discovery” stories.

# load dataset
# vectorize text with TF-IDF
# fit KMeans on X
# get cluster assignments
# evaluate with ARI and silhouette
# reduce X to 2D with TruncatedSVD
# project centroids with same SVD
# plot 2D points and projected centroids

# Since clustering operates in vector space, text must first be converted 
# into numerical representations such as TF-IDF or sentence embeddings 
# before applying clustering.

from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from sentence_transformers import SentenceTransformer

# load dataset
ds = load_dataset('banking77')

# split dataset
train = ds['train'].to_pandas()
test = ds['test'].to_pandas()
df = pd.concat([train, test])
print(df.head())

# TF-IDF vectorization
# tfidfvectorizer = TfidfVectorizer()
# X = tfidfvectorizer.fit_transform(df["text"])

# TF-IDF is lexical. Clustering usually improves with semantic embeddings.
model = SentenceTransformer("all-MiniLM-L6-v2")
X = model.encode(df["text"].tolist(), show_progress_bar=True)

# Initialize the KMeans estimator with the desired number of clusters (k=4)
kmeans = KMeans(n_clusters=40, random_state=42, n_init=10)

# Fit the model to the data and predict the cluster labels
y_kmeans = kmeans.fit_predict(X)

# TruncatedSVD
svd = TruncatedSVD(n_components=2, random_state=42)
X_2d = svd.fit_transform(X)
centers_2d = svd.transform(kmeans.cluster_centers_)

# Visualize the clusters and their centroids
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='black', s=200, alpha=0.5, marker='x')
plt.show()

# evaluate vs true labels
# Adjusted Rand Index measures how well your clusters align with the true labels.
ari = adjusted_rand_score(df['label'], y_kmeans)
print("Adjusted Rand Index:", ari)

# silhouette score (internal metric)
# Silhouette measures separation between clusters.
sil = silhouette_score(X, y_kmeans)
print("Silhouette:", sil)