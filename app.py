import streamlit as st
from utils import database as db
from services.kb import get_available_kbs, create_new_kb
from ui.sidebar import render_sidebar
from ui.chat import render_chat_page
from ui.settings import render_settings_page

# --- MAIN APP ---
st.set_page_config(page_title="SOP Assistant", layout="wide")
db.init_db()

if "session_initialized" not in st.session_state:
    kbs = get_available_kbs()
    st.session_state.update({
        "active_kb": kbs[0] if kbs else None, 
        "vectorstore": None, 
        "active_thread_id": None, 
        "view": "chat", 
        "session_initialized": True
    })

render_sidebar()

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