# BERTopic is useful because it combines semantic embeddings, 
# dimensionality reduction, clustering, and topic keyword extraction, 
# but for production or constrained environments I’d also consider a simpler 
# embeddings-plus-KMeans pipeline for stability and speed.

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

docs = fetch_20newsgroups(
    subset='all', 
    remove=('headers', 'footers', 'quotes')
)['data'][:300]

topic_model = BERTopic(
    embedding_model=embedding_model,
    verbose=True,
    calculate_probabilities=False
)
topics, probs = topic_model.fit_transform(docs)

topic_info = topic_model.get_topic_info()
print(topic_info.head())

# Explain topics
prompt = f"""
Explain the following discovered topics from a topic modeling pipeline.
Summarize the major themes and mention if some topics look noisy or overlapping.

Topic info:
{topic_info.to_string(index=False)}
"""
response = llm.invoke(prompt)
print(f"Topics info: {response.content}")