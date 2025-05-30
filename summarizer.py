"""
    Name: summarizer.py
    Purpose: Contains the langchain based summarizer logic
    Author: Shivani Shah
    Created At: 28th May,2025
"""

#Import necessary libraries
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
import torch

#Check if GPU is available
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device")


def load_document(file_path):
    """
        Purpose: Loading text from the documents
        Input: PDF File path or Word Document File path
        Output: returns loader for the text of the document
    """
    if file_path.endswith(".pdf"):
        loader=PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader=Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file type.Use pdf or docx")
    return loader.load()

def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    """
        Purpose: Convert the document text into chunks
        Input: documents, chunk_size, chunk_overlap
        Output returns chunks of text
    """
    splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def load_local_llm():
    """
        Purpose: Initialize the summarization pipeline
    """
    print("Loading the transformers summarization pipeline")
    model_name="facebook/bart-large-cnn"
    summarization_pipe=pipeline("summarization",model=model_name,device="cuda")
    print("Summarization Pipeline loaded")
    return HuggingFacePipeline(pipeline=summarization_pipe)

def summarize_documents(docs):
    """
        Purpose: Create the summary chain for the documents uploaded and generate summary
        Input: text of document
        Output: returns the summary generated
    """
    print("Loading LLM")
    llm=load_local_llm()
    print("Loading Summarization chain")
    chain=load_summarize_chain(llm, chain_type="refine")
    print("Invoking the chain")
    summary=chain.invoke(docs)
    print("Summary generated")
    return summary['output_text']

def summarize_function(file_path):
    """
        Purpose: Given the file path, generate the summary of the document
        Input: path of the file
        Output: returns the summary generated
    """
    print("Loading document")
    raw_docs=load_document(file_path)

    print("Splitting document")
    split_docs=split_documents(raw_docs)

    print("Summarizing...")
    summary=summarize_documents(split_docs)

    print("\n--- Summary ---\n")
    print(summary)
    return summary



