# classification pipeline
# TF-IDF + LogisticRegression -> TF-IDF + Linear SVM -> Sentence embeddings 
# + classifier -> Fine-tuned BERT
# Document Clustering/Classification: Used to convert text into 
# numerical vectors for machine learning models (e.g., spam detection).

# load
# clean train text
# clean test text
# fit vectorizer on train
# transform train
# transform test
# fit model on train
# predict on test
# evaluate

from datasets import load_dataset
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

lemmatizer = WordNetLemmatizer()

# load data
ds = load_dataset("banking77")
train = ds['train'].to_pandas()
test = ds['test'].to_pandas()
print(train.head())

# clean data
# def clean(text):
#     text = text.lower().strip()
#     text = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
#     tokens = word_tokenize(text)
#     lemmas = [lemmatizer.lemmatize(word) for word in tokens]
#     return " ".join(lemmas)

def clean(text):
    text = text.lower().strip()
    text = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    return " ".join(text.split())

train['text'] = train['text'].apply(clean)
test['text'] = test['text'].apply(clean)

# tf-idf for extracting features
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train["text"])
X_test = vectorizer.transform(test["text"])

# logistic regression
lr = LogisticRegression()
lr.fit(X_train, train['label'])

# evaluation
preds = lr.predict(X_test)
print("Macro F1:", f1_score(test["label"], preds, average="macro"))
label_names = ds["train"].features["label"].names
print(classification_report(test["label"], preds, target_names=label_names))