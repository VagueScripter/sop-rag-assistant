import os
import shutil
import re
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from config import KNOWLEDGE_BASE_DIR
from utils import database as db

def get_kb_path(kb_name):
    return os.path.join(KNOWLEDGE_BASE_DIR, kb_name)

def get_kb_documents_path(kb_name):
    return os.path.join(get_kb_path(kb_name), "documents")

def get_kb_index_path(kb_name):
    return os.path.join(get_kb_path(kb_name), "index")

def get_available_kbs():
    if os.path.exists(KNOWLEDGE_BASE_DIR):
        return sorted([d for d in os.listdir(KNOWLEDGE_BASE_DIR) if os.path.isdir(os.path.join(KNOWLEDGE_BASE_DIR, d))])
    return []

def extract_metadata_from_text(text: str):
    sop_number_match = re.search(r'SOP[ -]?(?:No|Number|#)?[: ]*([A-Za-z0-9-]+)', text, re.IGNORECASE)
    sop_number = sop_number_match.group(1) if sop_number_match else None
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    title = lines[0][:200] if lines else None
    return title, sop_number

def create_new_kb(kb_name):
    if " " in kb_name or not kb_name:
        st.error("Invalid name. Please avoid spaces.")
        return
    if os.path.exists(get_kb_path(kb_name)):
        st.warning(f"Knowledge Base '{kb_name}' already exists.")
    else:
        os.makedirs(get_kb_documents_path(kb_name), exist_ok=True)
        os.makedirs(get_kb_index_path(kb_name), exist_ok=True)
        st.success(f"Knowledge Base '{kb_name}' created.")
        st.session_state.active_kb = kb_name
        st.session_state.show_create_kb_form = False
        st.rerun()

def build_knowledge_base(kb_name, embeddings):
    doc_path = get_kb_documents_path(kb_name)
    index_path = get_kb_index_path(kb_name)
    with st.spinner(f"Building KB '{kb_name}'..."):
        loader = DirectoryLoader(doc_path, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=False)
        docs = loader.load()
        if not docs:
            st.warning(f"No PDF files found in '{kb_name}'.")
            if os.path.exists(index_path):
                shutil.rmtree(index_path)
            return None
            
        db.delete_parsed_documents(kb_name)
        
        for doc in docs:
            file_name = os.path.basename(doc.metadata.get("source", ""))
            page_number = doc.metadata.get("page", 0) + 1
            title, sop_number = extract_metadata_from_text(doc.page_content)
            db.save_parsed_document(kb_name, file_name, doc.page_content, title, sop_number, page_number)
            
            doc.metadata["title"] = title
            doc.metadata["sop_number"] = sop_number

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_docs = splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(final_docs, embeddings)
        vectorstore.save_local(index_path)
        st.success(f"KB '{kb_name}' built successfully! Parsed text stored for validation.")
    return vectorstore

def load_knowledge_base(kb_name, embeddings):
    index_path = get_kb_index_path(kb_name)
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
