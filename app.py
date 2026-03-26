import streamlit as st
import os
from dotenv import load_dotenv
import shutil
import json

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from utils import database as db

load_dotenv()

# --- CONFIGURATION ---
KNOWLEDGE_BASE_DIR = "knowledge_bases"

# --- SETUP & CACHING ---
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)

@st.cache_resource
def load_llm():
    return ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2, groq_api_key=os.getenv("GROQ_API_KEY"))

@st.cache_resource
def load_embeddings_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

llm = load_llm()
embeddings = load_embeddings_model()

# --- KNOWLEDGE BASE UTILS ---
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

def create_new_kb(kb_name):
    if " " in kb_name or not kb_name:
        st.error("Invalid name. Please avoid spaces.")
        return
    if os.path.exists(get_kb_path(kb_name)):
        st.warning(f"Knowledge Base '{kb_name}' already exists.")
    else:
        os.makedirs(get_kb_documents_path(kb_name))
        os.makedirs(get_kb_index_path(kb_name))
        st.success(f"Knowledge Base '{kb_name}' created.")
        st.session_state.active_kb = kb_name
        st.session_state.show_create_kb_form = False
        st.rerun()

def build_knowledge_base(kb_name):
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
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_docs = splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(final_docs, embeddings)
        vectorstore.save_local(index_path)
        st.success(f"KB '{kb_name}' built successfully!")
    return vectorstore

def load_knowledge_base(kb_name):
    index_path = get_kb_index_path(kb_name)
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

# --- PROMPTS (Restored to be strict) ---
system_prompt = (
    "You are an AI-powered Standard Operating Procedure (SOP) Assistant. "
    "Your goal is to provide accurate, concise, and safety-oriented information "
    "based strictly on the provided technical documents. "
    "\n\n"
    "GUIDELINES:"
    "1. If the information is not in the context, state: 'This information is not covered in the current SOP.' "
    "2. If the user asks about safety, prioritize caution and bold key warnings. "
    "3. Use bullet points for step-by-step instructions."
    "\n\n"
    "CONTEXT FROM SOP:"
    "\n{context}"
)
qa_prompt = ChatPromptTemplate.from_messages([("system", system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")])
contextualize_q_prompt = ChatPromptTemplate.from_messages([("system", "Given a chat history and a question, reformulate it into a standalone question."), MessagesPlaceholder("chat_history"), ("human", "{input}")])

# --- UTILITY ---
def generate_thread_title(user_query: str) -> str:
    title_prompt = ChatPromptTemplate.from_messages([("system", "Generate a concise title (max 5 words) for a conversation starting with this query:"), ("human", "{input}")])
    title_chain = title_prompt | llm
    return title_chain.invoke({"input": user_query}).content.strip().strip('"')

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- UIRENDERING ---
def render_settings_page():
    os.makedirs(get_kb_documents_path(st.session_state.active_kb), exist_ok=True)
    os.makedirs(get_kb_index_path(st.session_state.active_kb), exist_ok=True)

    if "editing_kb_name" not in st.session_state:
        st.session_state.editing_kb_name = False

    st.header(f"Manage '{st.session_state.active_kb}'")
    if st.button("⬅️ Back to Chat"):
        st.session_state.view = "chat"
        st.rerun()
    st.divider()
    
    st.subheader("Rename Knowledge Base")
    # ... (rest of the settings page logic)
    if not st.session_state.editing_kb_name:
        if st.button("✏️ Rename"):
            st.session_state.editing_kb_name = True
            st.rerun()
    else:
        new_kb_name_edit = st.text_input("New Name", value=st.session_state.active_kb, key="edit_kb_name_input")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Save"):
                if new_kb_name_edit and " " not in new_kb_name_edit and new_kb_name_edit not in get_available_kbs():
                    os.rename(get_kb_path(st.session_state.active_kb), get_kb_path(new_kb_name_edit))
                    db.rename_kb(st.session_state.active_kb, new_kb_name_edit)
                    st.session_state.active_kb = new_kb_name_edit
                    st.session_state.editing_kb_name = False
                    st.rerun()
                else: st.error("Invalid or existing name.")
        with col2:
            if st.button("❌ Cancel"):
                st.session_state.editing_kb_name = False
                st.rerun()
    st.divider()

    st.subheader("Knowledge Base Contents")
    uploaded_files = st.file_uploader("Upload new documents", type="pdf", accept_multiple_files=True, key=f"uploader_{st.session_state.active_kb}")
    if uploaded_files:
        for file in uploaded_files:
            with open(os.path.join(get_kb_documents_path(st.session_state.active_kb), file.name), "wb") as f:
                f.write(file.getbuffer())
        st.success(f"{len(uploaded_files)} file(s) saved. Rebuild required.")
    
    if st.button("Rebuild Knowledge Base"):
        st.session_state.vectorstore = build_knowledge_base(st.session_state.active_kb)
    st.divider()
    
    st.subheader("Create New Knowledge Base")
    with st.form("new_kb_form", clear_on_submit=True):
        new_kb_name = st.text_input("New KB Name (no spaces)")
        if st.form_submit_button("Create"):
            create_new_kb(new_kb_name)
    st.divider()

    st.subheader("Danger Zone")
    delete_confirmation = st.text_input(f"To confirm deletion, type the KB name: **{st.session_state.active_kb}**")
    if st.button("Delete This Knowledge Base", type="primary", disabled=(delete_confirmation != st.session_state.active_kb)):
        shutil.rmtree(get_kb_path(st.session_state.active_kb))
        db.delete_kb_threads(st.session_state.active_kb)
        st.session_state.active_kb = None
        st.session_state.view = "chat"
        st.rerun()

def render_chat_page():
    if st.session_state.active_kb and st.session_state.get("vectorstore") is None:
        index_path = get_kb_index_path(st.session_state.active_kb)
        if os.path.exists(index_path) and os.listdir(index_path):
             with st.spinner(f"Loading KB '{st.session_state.active_kb}'..."):
                st.session_state.vectorstore = load_knowledge_base(st.session_state.active_kb)
        else:
            st.warning(f"KB '{st.session_state.active_kb}' is empty. Go to settings ⚙️ to upload and build.")
    
    if st.session_state.active_thread_id:
        st.header(db.get_thread_title(st.session_state.active_thread_id))
        chat_history = db.get_messages_by_thread(st.session_state.active_thread_id)
        for msg in chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["sources"]: st.caption(f"Sources: {json.loads(msg['sources'])}")
    else:
        st.header(f"Chat with '{st.session_state.active_kb}'")

    if user_query := st.chat_input("Ask a question..."):
        if st.session_state.vectorstore is None:
            st.error("Knowledge base not loaded. Please build it in settings ⚙️.")
        else:
            if st.session_state.active_thread_id is None:
                new_title = generate_thread_title(user_query)
                st.session_state.active_thread_id = db.create_new_thread(new_title, st.session_state.active_kb)
            db.add_message_to_thread(st.session_state.active_thread_id, "user", user_query)
            with st.chat_message("assistant"):
                with st.spinner("Searching..."):
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
                    h_a_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
                    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
                    rag_chain = create_retrieval_chain(h_a_retriever, qa_chain)
                    history = [HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"]) for m in db.get_messages_by_thread(st.session_state.active_thread_id)]
                    response = rag_chain.invoke({"input": user_query, "chat_history": history})
                    sources = list(set(os.path.basename(doc.metadata.get("source", "")) for doc in response.get("context", [])))
                    db.add_message_to_thread(st.session_state.active_thread_id, "assistant", response["answer"], sources=json.dumps(sources) if sources else None)
            st.rerun()

# --- MAIN APP ---
st.set_page_config(page_title="SOP Assistant", page_icon="🤖", layout="wide")
db.init_db()

if "session_initialized" not in st.session_state:
    kbs = get_available_kbs()
    st.session_state.update({"active_kb": kbs[0] if kbs else None, "vectorstore": None, "active_thread_id": None, "view": "chat", "session_initialized": True})

with st.sidebar:
    load_css("style.css")
    st.title("SOP Bot")
    selected_kb = st.selectbox("Select Knowledge Base", get_available_kbs(), key="kb_selector")
    if selected_kb and selected_kb != st.session_state.active_kb:
        st.session_state.update({"active_kb": selected_kb, "vectorstore": None, "active_thread_id": None, "view": "chat"})
        st.rerun()
    
    st.divider()
    
    if st.session_state.active_kb:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ New Chat"):
                st.session_state.active_thread_id = None
                st.session_state.view = "chat"
                st.rerun()
        with col2:
            if st.button("Settings ⚙️"):
                st.session_state.view = "settings"
                st.rerun()
        
        st.divider()
        st.header("Chat History")
        threads = db.get_all_threads(st.session_state.active_kb)
        for thread in threads:
            if st.button(thread['title'], key=f"thread_{thread['id']}"):
                if st.session_state.active_thread_id != thread['id']:
                    st.session_state.active_thread_id = thread['id']
                    st.session_state.view = "chat"
                    st.rerun()

# --- MAIN PANEL ---
if st.session_state.get("view") == "settings" and st.session_state.active_kb:
    render_settings_page()
elif st.session_state.active_kb:
    render_chat_page()
else:
    st.title("SOP Intelligence Bot")
    st.info("Welcome! Please create your first Knowledge Base to get started.")
    with st.form("initial_kb_form"):
        new_kb_name = st.text_input("Enter a name for your first Knowledge Base")
        if st.form_submit_button("Create"):
            create_new_kb(new_kb_name)