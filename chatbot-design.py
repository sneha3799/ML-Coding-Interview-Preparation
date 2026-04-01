# Capital One Chatbot Design 

# Let’s say we’re building a customer support chatbot for Capital One’s 
# online banking platform, and we have access to customer account details, 
# transaction histories, and common support FAQs.
# How would you approach designing this chatbot to ensure it provides secure, 
# helpful, and relevant responses to customers?

# store transaction histories and account details (structured data) in postgresql
# load customer support FAQs -> chunking -> add metadata -> metadata filtering
# When the query is related to transaction history or account details convert text
# to sql and fetch data from postgresql
# else retrieve from vectorstore and let LLM generate response

# Recommended architecture

# Offline
# load and chunk FAQs / policies
# embed and index in vector DB
# attach metadata
# maintain structured banking data in PostgreSQL or service APIs

# Online
# authenticate user
# classify intent
# if account-specific → call backend tool(s)
# if FAQ/support → retrieve via RAG
# compose grounded response
# log and monitor

import os
import json
import psycopg2
import xgboost as xgb
from dotenv import load_dotenv
load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

from sentence_transformers import CrossEncoder
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

embedding_fn = HuggingFaceEmbeddings(model_name = "sentence_transformers/all-MiniLM-L6-v2")

# load docs
docs = PyPDFLoader("test.pdf").load()

# split docs
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = splitter.split_documents(docs)

# add metadata
for i, doc in enumerate(chunks):
    doc.metadata["chunk_id"] = f'chunk_{i}'
    doc.metadata["source"] = "test.pdf"
    doc.metadata["access_users"] = ["risk", "support"]
    doc.metadata["doc_type"] = "faq"

# hybrid retrieval
db = FAISS.from_documents(chunks, embedding_fn)
db.save_local("faiss_index")

dense_retriever = db.as_retriever(search_kwargs={"k": 3})
sparse_retriever = BM25Retriever.from_documents(chunks)
sparse_retriever.k = 3

def deduplication_by_chunk_id(all_chunks):
    seen = set()
    result = []
    for doc in all_chunks:
        cid = doc.metadata["chunk_id"]
        if cid not in seen:
            seen.add(cid)
            result.append(doc)
    return result

def hybrid_retriever(query):
    dense_docs = dense_retriever.invoke(query)
    sparse_docs = sparse_retriever.invoke(query)
    all_docs = dense_docs + sparse_docs
    deduped = deduplication_by_chunk_id(all_docs)
    return deduped

# permission filtering
def filter_by_permissions(docs, user):
    filtered = []
    for doc in docs:
        if user in doc.metadata["access_users"]:
            filtered.append(doc)
    return filtered

# reranking
def reranking(query, docs, top_k=3):
    pairs = [[query, doc.page_content] for doc in docs]
    scores = ce.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for score, doc in ranked[:top_k]]

url = os.getenv('URL')
def get_last_transaction():
    conn = psycopg2.connect(url)
    cur = conn.cursor()
    results = []
    try:
        cur.execute("""
            SELECT id, name, amount
            FROM transactions
            ORDER BY date DESC
            LIMIT 1
        """)
        results = cur.fetchall()
    except Exception as e:
        print(f"Exception: {e}")
    finally:
        cur.close()
        conn.close()
    return results
    

def route_query(query: str) -> str:
    q = query.lower()
    if "transaction" in q or "balance" in q or "payment" in q:
        return "transactions"
    return "faq"

client = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

query = "How do I reset pwd?"
query_type = route_query(query)

if query_type == "transactions":
    data = get_last_transaction()

    prompt = f"""
    You are a banking assistant.
    Explain the following transaction result clearly to the user:

    {data}
    """

    response = client.invoke(prompt)
    print("Transaction:", response.content)

elif query_type == "faq":
    retrieved = hybrid_retriever(query)
    filtered = filter_by_permissions(retrieved, "support")
    top_docs = reranking(query, filtered, top_k=3)

    context = "\n\n".join(
        f"[{doc.metadata['source']} | {doc.metadata['chunk_id']}]\n{doc.page_content}"
        for doc in top_docs
    )

    prompt = f"""
    You are a helpful banking assistant.
    Answer the user's question using only the provided FAQ context.
    If the answer is not in the context, say you don't know.

    Context:
    {context}

    Question:
    {query}
    """

    response = client.invoke(prompt)
    print("FAQ:", response.content)