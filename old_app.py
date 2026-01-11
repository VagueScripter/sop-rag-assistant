import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

#v2 header changes below
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
# Load API Key
load_dotenv()

# Streamlit Page Config
st.set_page_config(page_title="SOP Assistant", page_icon="🍔")
st.title("McDonald's SOP Intelligence Bot")
st.markdown("Upload any SOP (Standard Operating Procedure) and ask questions.")

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# 1. File Upload Logic
uploaded_file = st.sidebar.file_uploader("Upload SOP (PDF)", type="pdf")

if uploaded_file:
    # Save the file temporarily in WSL
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # 2. Process the Document
    loader = PyPDFLoader("temp.pdf")
    data = loader.load()
    
    # Split text into chunks so the AI doesn't get overwhelmed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(data)
    
    # 3. Create Vector Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    st.sidebar.success("SOP Processed Successfully!")

    # 4. Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("How do I clean the grill?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # RAG Logic: Retrieve and Answer
        with st.chat_message("assistant"):
            # Define how the AI should behave
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
            
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            # Combine everything into a chain
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(vector_store.as_retriever(), question_answer_chain)
            
            # Run the chain
            response = rag_chain.invoke({"input": prompt})
            answer = response["answer"]
            
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
