# Case Study: Generative AI for Fraud Detection
# CIBC's fraud operations team currently receives 2,000 flagged transactions per day. For each flag, an analyst must manually review the transaction details, pull the customer's recent history, check it against fraud patterns, and write a brief risk assessment with a recommended action (escalate, block, or release).
# This process takes 8–12 minutes per case. The team wants to use GenAI to automatically generate a structured risk assessment for each flagged transaction that includes: a summary of why the transaction was flagged, relevant context from the customer's history, a risk score justification, and a recommended action — all grounded in the transaction data and internal fraud rules.
# You're given:
# A CSV of flagged transactions with fields: txn_id, customer_id, amount, merchant, category, location, timestamp, customer_avg_txn, days_since_last_txn, is_international, flag_reason
# A set of internal fraud rule documents (PDFs) that define thresholds, patterns, and escalation criteria

import os
import json
import psycopg2

from dotenv import load_dotenv
load_dotenv()

from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_core.prompts import PromptTemplate

from pydantic import BaseModel, Field

# Output schema
class RiskAssessment(BaseModel):
    flagged_summary: str = Field(description="Why the transaction was flagged")
    customer_history_context: str = Field(description="Relevant context from recent customer history")
    risk_justification: str = Field(description="Why the risk score/recommendation makes sense")
    recommended_action: str = Field(description="One of: escalate, block, release")

ce = CrossEncoder(model_name_or_path="ms-marco-MiniLM-L6-v2")
embedding_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# load docs
thresholds = PyPDFLoader("thresholds.pdf").load()
patterns = PyPDFLoader("patterns.pdf").load()
escalation_criteria = PyPDFLoader("Escalation_criteria.pdf").load()

# add metadata
for doc in thresholds+patterns:
    doc.metadata["access_users"] = ["risk", "support"]
    doc.metadata["source"] = "thresholds.pdf"
    doc.metadata["doc_type"] = "faqs"

for doc in escalation_criteria:
    doc.metadata["source"] = "escalation_criteria.pdf"
    doc.metadata["doc_type"] = "policy"
    doc.metadata["access_users"] = "risk"

# split docs
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
semantic_splitter = SemanticChunker(
    embeddings=embedding_fn,
    breakpoint_threshold_type="percentile"
)
all_chunks = text_splitter.split_documents(thresholds)
all_chunks.extend(text_splitter.split_documents(patterns))
all_chunks.extend(semantic_splitter.split_documents(escalation_criteria))

for i, doc in enumerate(all_chunks):
    doc.metadata["chunk_id"] = f"chunk_{i}"

# embed + store
db = FAISS.from_documents(all_chunks, embedding_fn)
db.save_local("faiss_local")
print(db.index.ntotal)

# retrieve
dense_retriever = db.as_retriever(search_kwargs={"k":3})
sparse_retriever = BM25Retriever.from_documents(all_chunks)
sparse_retriever.k = 3

def dedup_by_chunk_id(all_chunks):
    seen = set()
    results = []
    for doc in all_chunks:
        cid = doc.metadata["chunk_id"]
        if cid not in seen:
            seen.add(cid)
            results.append(doc)
    return results

def hybrid_retriever(query):
    dense_docs = dense_retriever.invoke(query)
    sparse_docs = sparse_retriever.invoke(query)
    all_docs = dense_docs + sparse_docs
    deduped = dedup_by_chunk_id(all_docs)
    return deduped
    
# permission filtering
def filtered(query, user):
    chunks = hybrid_retriever(query)
    filtered = []
    for chunk in chunks:
        if user in chunk.metadata["access_users"]:
            filtered.append(chunk)
    return filtered

# reranking
def reranked(query, user, top_k=3):
    docs = filtered(query, user)
    pairs = [[query, doc.page_content] for doc in docs]
    scores = ce.predict(pairs)
    rerank_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return " ".join([passage.page_content for score, passage in rerank_docs][:top_k])

# generate
def policy(query):
    prompt_template = PromptTemplate.from_template("generate fraud criteria based on the provided docs {context}")
    prompt = prompt_template.format(context = reranked(query, "risk"))

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    conditions = llm.invoke(prompt)
    return conditions

# tool to access customer history
def customer_history(id):
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id, amount, time, flag_reason
            FROM Transactions
            FILTER BY id
            ORDER BY date DESC
            LIMIT 1
        """)
        results = cur.fetchall()
        print(f"Transaction details: {results}")
    except Exception as e:
        print(f"Exception : {e}")
    finally:
        cur.close()
        conn.close()
    return results

# router
def router(query, id):
    transactions = customer_history(id)
    conditions = policy(query)

    prompt_template = PromptTemplate.from_template("Generate a risk assessment based on customer history {transactions} and policy {conditions}")
    prompt = prompt_template.format(transactions=transactions, conditions=conditions)

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    structured_llm = llm.with_structured_output(RiskAssessment)  

    results = structured_llm.invoke(prompt)
    return results.content.strip()

results = router("Generate a risk assessment for the flagged transaction?")
print(f"Results: {results}")

