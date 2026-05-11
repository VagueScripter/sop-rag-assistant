import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config import GROQ_API_KEY, MODEL_NAME, EMBEDDINGS_MODEL

@st.cache_resource
def load_llm():
    return ChatGroq(model_name=MODEL_NAME, temperature=0.2, groq_api_key=GROQ_API_KEY)

@st.cache_resource
def load_embeddings_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL, model_kwargs={"device": "cpu"})

def get_system_prompt():
    return (
        "You are an AI-powered Standard Operating Procedure (SOP) Assistant. "
        "Your goal is to provide accurate, concise, and safety-oriented information "
        "based strictly on the provided technical documents. "
        "\n\n"
        "GUIDELINES:\n"
        "1. If the information to answer the question is not in the context, state EXACTLY: 'I don't have an answer to that question because it is not covered in the knowledge base.' Do not use outside knowledge.\n"
        "2. If the user asks about safety, prioritize caution and bold key warnings.\n"
        "3. Use bullet points for step-by-step instructions.\n"
        "\n\n"
        "CONTEXT FROM SOP:\n"
        "{context}"
    )

def get_qa_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", get_system_prompt()), 
        MessagesPlaceholder("chat_history"), 
        ("human", "{input}")
    ])

def get_contextualize_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and a question, reformulate it into a standalone question."), 
        MessagesPlaceholder("chat_history"), 
        ("human", "{input}")
    ])

def generate_thread_title(llm, user_query: str) -> str:
    title_prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate a concise title (max 5 words) for a conversation starting with this query:"), 
        ("human", "{input}")
    ])
    title_chain = title_prompt | llm
    return title_chain.invoke({"input": user_query}).content.strip().strip('"')
