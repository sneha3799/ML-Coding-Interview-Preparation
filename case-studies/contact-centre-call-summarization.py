# Case Study A: Contact Centre Call Summarization
# CIBC's contact centre handles 50,000 calls per day. After each call, agents spend 3–5 minutes writing notes and 
# updating the CRM. You're given call transcripts (text files, average 2,000 words each) and a set of internal CRM 
# field definitions (PDFs) that specify what fields need to be filled: reason for call, resolution, follow-up 
# actions, customer sentiment, and escalation flag. Build a system that takes a transcript and generates structured 
# CRM-ready notes.

import os
import json

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from typing import Literal
import pii_masker
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever


# Structured output
class FieldDefinitions(BaseModel):
    reason: str = Field(description="Reason for call")
    resolution: str = Field(description="Resolution provided to the customer")
    actions: list[str] = Field(description="Any follow-up actions")
    sentiment: Literal["positive", "neutral", "negative", "frustrated"] = Field(description="Customer sentiment")
    flag: bool  = Field(description="Escalation flag, mark as yes or no")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
structured_llm = llm.with_structured_output(FieldDefinitions)

# PII masking
def preprocessing(transcript):
    text = pii_masker.mask_pii(transcript)
    return text

def RAG(query):
    # load docs
    # add metadata
    # split docs
    # embed + store
    # hybrid retrieval
    # permission filtering
    # reranking
    # generate what fields needs to be filled
    return ""


text = preprocessing("I am having issues accessing the web application. It's frustrating. I need to check it asap.")
context = RAG(text)

# Generate CRM-ready notes
prompt_template = PromptTemplate.from_template(
    "Based on the provided transcript and attached context generate CRM ready notes " \
    "Transcript: {transcript}" \
    "Context: {context}"
)
prompt = prompt_template.format(transcript=text, context=context)
output = structured_llm.invoke(prompt)
print(output)

# for each transcript in transcript_dir:
#     clean and mask PII
#     generate structured CRM notes
#     append to results



# from langchain_text_splitters import RecursiveCharacterTextSplitter
# splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
# chunks = splitter.split_text(transcript)

# # Map: summarize each chunk
# chunk_summaries = []
# for chunk in chunks:
#     summary = llm.invoke(f"Summarize this call segment:\n{chunk}")
#     chunk_summaries.append(summary.content)

# # Reduce: generate final structured output from all summaries
# combined = "\n\n".join(chunk_summaries)
# final_output = structured_llm.invoke(
#     prompt.format(transcript=combined, context=context)
# )