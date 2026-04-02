# Case Study B: Internal Policy Q&A for Compliance Officers
# CIBC's compliance team needs to answer regulatory questions by searching across 500+ internal policy documents, 
# regulatory bulletins, and past audit reports. An officer currently spends 30 minutes per question gathering 
# relevant sections. Build a system where an officer types a question like "What are our obligations under FINTRAC 
# for transactions over $10,000?" and gets a grounded, cited answer. The system must track which source documents 
# were used and flag when confidence is low.

# RAG system
# structured output (source docs used, flag, confidence)

import os

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate

from sentence_transformers import CrossEncoder

embedding_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
ce = CrossEncoder(model_name_or_path="ms-marco-miniLM-L6-v2")

class Output(BaseModel):
    confidence: Literal["High", "Low", "Medium"] = Field(description="How confident is the model about the answer?")
    flag: bool = Field(description="Set to True or False")
    source_docs: list[str] = Field(description="Sources documents that were used")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
structured_llm = llm.with_structured_output(Output)

# load docs
docs = PyPDFDirectoryLoader("/company-docs/", "*.pdf").load()

# add metadata
for doc in docs:
    doc.metadata["source"] = doc.metadata.get("source")
    doc.metadata["access_users"] = ["compliance officer"]
    doc.metadata["doc_type"] = "company_policy"

# split docs
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
all_chunks = text_splitter.split_documents(docs)
for i, doc in enumerate(all_chunks):
    doc.metadata["chunk_id"] = f"chunk_{i}"

# embed + store
db = FAISS.from_documents(all_chunks, embedding_fn)
db.save_local("faiss_index")

# hybrid retrieval
dense_retriever = db.as_retriever(search_kwargs={"k":3})
sparse_retriever = BM25Retriever.from_documents(all_chunks)
sparse_retriever.k = 3

def deduped_by_chunk_id(chunks):
    seen = set()
    results = []
    for chunk in chunks:
        cid = chunk.metadata["chunk_id"]
        if cid not in seen:
            seen.add(cid)
            results.append(chunk)
    return results

def hybrid_retriever(query):
    dense_docs = dense_retriever.invoke(query)
    sparse_docs = sparse_retriever.invoke(query)
    combined = dense_docs + sparse_docs
    deduped = deduped_by_chunk_id(combined)
    return deduped
    
# permission filtering
def filtered_chunks(query, user):
    filtered = []
    docs = hybrid_retriever(query)
    for doc in docs:
        if user in doc.metadata["access_users"]:
            filtered.append(doc)
    return filtered

# rerank
def rerank(query, user, top_k=3):
    filtered = filtered_chunks(query, user)
    pairs = [[query, doc.page_content] for doc in filtered]
    scores = ce.predict(pairs)
    reranked_passages = sorted(zip(scores, filtered), key=lambda x: x[0], reverse=True)
    return " ".join([passage.page_content for score, passage in reranked_passages][:top_k])

# generate
prompt_template = PromptTemplate.from_template("You are a helpful assistant. Your role is to analyze the " \
"given documents and then provide a grounded, cited answer for the query. Also set flag to True if confidence is low." \
"documents: {context}" \
"query: {ques}"
)

ques = "What are our obligations under FINTRAC for transactions over $10,000?"
docs = rerank(ques, "compliance officer")
prompt = prompt_template.format(context=docs, ques=ques)
output = structured_llm.invoke(prompt)
print(f"Output: {output}")