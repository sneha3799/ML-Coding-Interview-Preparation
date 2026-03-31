# RAG pipeline
# load docs → chunk → attach metadata/permissions → build BM25 retriever → 
# build vector retriever → hybrid retrieve → permission filter → 
# rerank → generate

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import CrossEncoder
# Load a pre-trained cross-encoder model
# A popular model fine-tuned for reranking is "cross-encoder/ms-marco-MiniLM-L6-v2"
model_name = 'cross-encoder/ms-marco-MiniLM-L6-v2'
reranker = CrossEncoder(model_name)

embedding_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

ques = input('Enter your query: ')
llm = ChatOpenAI(model='gpt-3.5-turbo', api_key=os.getenv('OPENAI_API_KEY'))

# 'all-MiniLM-L6-v2' is a fast and effective model.

# load docs
policy_docs = PyPDFLoader("internal-policy.pdf").load()
qa_docs = PyPDFLoader("QA.pdf").load()

# add metadata
for doc in policy_docs:
    doc.metadata["source"] = "internal-policy.pdf"
    doc.metadata["doc_type"] = "policy"
    doc.metadata["access_groups"] = ["risk", "support"]
for doc in qa_docs:
    doc.metadata["source"] = "QA.pdf"
    doc.metadata["doc_type"] = "faq"
    doc.metadata["access_groups"] = ["support"]

# chunking
# high-volume docs: fast recursive splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)
# high-value docs: semantic splitting
semantic_splitter = SemanticChunker(
    embedding_fn,
    breakpoint_threshold_type="percentile",
)
# flatten into single list
all_chunks = text_splitter.split_documents(policy_docs)
all_chunks.extend(semantic_splitter.split_documents(qa_docs))
for i, doc in enumerate(all_chunks):
    doc.metadata["chunk_id"] = f"chunk_{i}"
    
# embed and store 
db = FAISS.from_documents(all_chunks, embedding_fn)
print(f"FAISS vector store contains {db.index.ntotal} vectors.")
db.save_local("faiss_index")

# retrieve
dense_retriever = db.as_retriever(search_kwargs={"k": 3})

bm25_retriever = BM25Retriever.from_documents(all_chunks)
bm25_retriever.k = 3

def deduplicate_by_chunk_id(docs):
    seen = set()
    result = []
    for doc in docs:
        cid = doc.metadata.get("chunk_id")
        if cid not in seen:
            seen.add(cid)
            result.append(doc)
    return result

def hybrid_retrieve(query):
    dense_docs = dense_retriever.invoke(query)
    sparse_docs = bm25_retriever.invoke(query)
    combined = dense_docs + sparse_docs
    deduped = deduplicate_by_chunk_id(combined)
    return deduped

# permission filtering
def filter_by_permissions(docs, user_groups):
    filtered = []
    for doc in docs:
        allowed_groups = set(doc.metadata.get("access_groups", []))
        if allowed_groups & user_groups:
            filtered.append(doc)
    return filtered

user_groups = {"support"}
authorized_docs = filter_by_permissions(hybrid_retrieve(ques), user_groups)
# if not authorized_docs:
#     print("No accessible documents found for this user.")
    

# reranking
# Pair the query with each passage to create model inputs
# Cross-encoders take (query, document) pairs as input
query_passage_pairs = [[ques, doc.page_content] for doc in authorized_docs]
# Score the pairs using the cross-encoder model
# The model outputs a single relevance score for each pair
scores = reranker.predict(query_passage_pairs)
# Sort the passages based on their relevance scores
# Combine scores and passages, then sort in descending order of score
ranked_passages = sorted(zip(scores, authorized_docs), key=lambda x: x[0], reverse=True)
passages = [passage for score, passage in ranked_passages][:3]

# generate
context = "\n\n".join(
    f"[{doc.metadata.get('source')} | {doc.metadata.get('chunk_id')}]\n{doc.page_content}"
    for doc in passages
)
prompt = PromptTemplate.from_template("""
    Answer the question {ques} using the context {context}
""")
prompt = prompt.format(ques=ques, context=context)
response = llm.invoke(prompt)
print(f'Response: {response.content}')

# evaluate
eval_queries = [
    {
        "query": "How do I reset my password?",
        "gold_chunk_id": "chunk_3"
    },
    {
        "query": "How do I dispute a charge?",
        "gold_chunk_id": "chunk_3"
    }
]

def hit_at_k(retrieved_docs, gold_chunk_id, k=3):
    retrieved_ids = [doc.metadata["chunk_id"] for doc in retrieved_docs[:k]]
    return int(gold_chunk_id in retrieved_ids)

def mrr(retrieved_docs, gold_chunk_id):
    retrieved_ids = [doc.metadata["chunk_id"] for doc in retrieved_docs]
    for i, chunk_id in enumerate(retrieved_ids):
        if chunk_id == gold_chunk_id:
            return 1 / (i + 1)
    return 0

hits = []
mrrs = []

for item in eval_queries:
    retrieved_docs = hybrid_retrieve(item["query"])
    authorized_docs = filter_by_permissions(retrieved_docs, {"support"})

    hits.append(hit_at_k(authorized_docs, item["gold_chunk_id"], k=3))
    mrrs.append(mrr(authorized_docs, item["gold_chunk_id"]))

print("Hit@3:", sum(hits) / len(hits))
print("MRR:", sum(mrrs) / len(mrrs))