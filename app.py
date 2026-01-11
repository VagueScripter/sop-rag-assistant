import streamlit as st
import os
from dotenv import load_dotenv

# 1. LLM & Embeddings
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# 2. Document Loading & Splitting
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 3. Vector Storage
from langchain_community.vectorstores import FAISS

# 4. Chains
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# --- CONFIGURATION ---
DATA_DIR = os.getenv("SOP_DATA_DIR", "data")
INDEX_DIR = os.getenv("SOP_INDEX_DIR", "faiss_index")

# --- SETUP ---
# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize LLM and Embeddings
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.2,
    groq_api_key=os.getenv("GROQ_API_KEY")
)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# Define a single, reusable prompt template
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
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# --- KNOWLEDGE BASE FUNCTIONS ---
def build_knowledge_base():
    """Builds the vectorstore from PDFs in the data directory and saves it."""
    with st.spinner("Building new knowledge base from PDF files..."):
        loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
        docs = loader.load()
        if not docs:
            st.warning("No PDF files found in the data directory. The knowledge base is empty.")
            return None
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_docs = splitter.split_documents(docs)
        
        vectorstore = FAISS.from_documents(final_docs, embeddings)
        vectorstore.save_local(INDEX_DIR)
        st.sidebar.success("Knowledge base built successfully!")
    return vectorstore

def load_knowledge_base():
    """Loads an existing vectorstore from the index directory."""
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

import database as db

# --- STREAMLIT APP ---
st.set_page_config(page_title="SOP Assistant", page_icon="🍔", layout="wide")
st.title("SOP Intelligence Bot")

# --- DATABASE & SESSION STATE INITIALIZATION ---
db.init_db()

# Initialize session state keys
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "active_thread_id" not in st.session_state:
    threads = db.get_all_threads()
    # Set the first thread as active if it exists, otherwise None
    st.session_state.active_thread_id = threads[0]["id"] if threads else None


# --- SIDEBAR ---
with st.sidebar:
    st.header("Chat Threads")

    if st.button("➕ New Chat"):
        st.session_state.active_thread_id = None
        # No rerun needed, Streamlit handles it
        
    st.divider()

    # Display chat threads
    threads = db.get_all_threads()
    for thread in threads:
        thread_id = thread["id"]
        # Use a more descriptive button label, perhaps with a short title
        button_label = thread['title']
        if st.button(button_label, key=f"thread_{thread_id}"):
            st.session_state.active_thread_id = thread_id
            # No rerun needed, Streamlit handles it
    
    st.divider()

    # Knowledge Base Management section
    with st.expander("📁 Knowledge Base Management"):
        st.caption(f"Current Archive: {os.path.abspath(DATA_DIR)}")

        # Auto-initialization logic for the vectorstore
        if st.session_state.vectorstore is None:
            if os.path.exists(INDEX_DIR):
                with st.spinner("Loading existing knowledge base..."):
                    st.session_state.vectorstore = load_knowledge_base()
                    st.success("Knowledge base loaded.")
            else:
                st.info("No existing knowledge base found. Building a new one...")
                st.session_state.vectorstore = build_knowledge_base()
        
        # Force a rebuild of the knowledge base
        if st.button("Rebuild Knowledge Base"):
            st.session_state.vectorstore = build_knowledge_base()

        # File uploader to add new documents
        uploaded_file = st.file_uploader("Add a new SOP (PDF)", type="pdf")
        if uploaded_file:
            with st.spinner(f"Processing '{uploaded_file.name}'..."):
                file_path = os.path.join(DATA_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Rebuild the entire knowledge base to include the new file
                st.session_state.vectorstore = build_knowledge_base()
                st.success(f"'{uploaded_file.name}' added and KB updated.")

# --- UTILITY FUNCTION ---
def generate_thread_title(user_query: str) -> str:
    """Generates a concise title for a new chat thread using the LLM."""
    title_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at creating short, concise titles for conversations. "
                   "Generate a title (max 5 words) for a conversation that starts with this user query:"),
        ("human", "{input}")
    ])
    title_chain = title_prompt | llm
    response = title_chain.invoke({"input": user_query})
    # Clean up the response content
    return response.content.strip().strip('"')


# --- CHAT INTERFACE ---
if st.session_state.active_thread_id:
    st.header(db.get_thread_title(st.session_state.active_thread_id))
    # Display chat history
    chat_history_from_db = db.get_messages_by_thread(st.session_state.active_thread_id)
    for message in chat_history_from_db:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
else:
    st.header("New Chat")
    st.info("Ask a question to start a new chat thread.")

# Main chat input
if user_query := st.chat_input("Ask a question about your SOPs..."):
    # If this is the first message in a new chat, create the thread first
    if st.session_state.active_thread_id is None:
        with st.spinner("Generating title..."):
            new_title = generate_thread_title(user_query)
        st.session_state.active_thread_id = db.create_new_thread(new_title)

    # Add user message to DB and display it
    db.add_message_to_thread(st.session_state.active_thread_id, "user", user_query)
    with st.chat_message("user"):
        st.markdown(user_query)
        
    # Get assistant response
    if st.session_state.vectorstore is not None:
        with st.chat_message("assistant"):
            with st.spinner("Searching through manuals..."):
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})

                history_aware_retriever = create_history_aware_retriever(
                    llm, retriever, contextualize_q_prompt
                )
                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

                # Fetch history from DB for the chain
                chat_history_for_chain = [
                    HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
                    for msg in db.get_messages_by_thread(st.session_state.active_thread_id)
                ]
                
                response = rag_chain.invoke({
                    "input": user_query,
                    "chat_history": chat_history_for_chain
                })
                answer = response["answer"]
                
                # Add assistant message to DB and display it
                db.add_message_to_thread(st.session_state.active_thread_id, "assistant", answer)
                st.markdown(answer)
                
                # Show sources
                sources = set(doc.metadata.get("source", "Unknown") for doc in response["context"])
                if sources:
                    st.caption(f"Sources consulted: {', '.join(sources)}")
    else:
        st.error("The knowledge base is not loaded. Please check the data folder or upload a file.")

