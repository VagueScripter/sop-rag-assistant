import os
import shutil
import streamlit as st
from utils import database as db
from services.kb import (
    get_kb_path, get_kb_documents_path, get_kb_index_path,
    get_available_kbs, create_new_kb, build_knowledge_base
)
from services.llm import load_embeddings_model

def render_settings_page():
    os.makedirs(get_kb_documents_path(st.session_state.active_kb), exist_ok=True)
    os.makedirs(get_kb_index_path(st.session_state.active_kb), exist_ok=True)

    if "editing_kb_name" not in st.session_state:
        st.session_state.editing_kb_name = False

    st.header(f"Manage '{st.session_state.active_kb}'")
    if st.button("Back to Chat", icon=":material/arrow_back:"):
        st.session_state.view = "chat"
        st.rerun()
    st.divider()
    
    st.subheader("Rename Knowledge Base")
    if not st.session_state.editing_kb_name:
        if st.button("Rename", icon=":material/edit:"):
            st.session_state.editing_kb_name = True
            st.rerun()
    else:
        new_kb_name_edit = st.text_input("New Name", value=st.session_state.active_kb, key="edit_kb_name_input")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save", icon=":material/check:"):
                if new_kb_name_edit and " " not in new_kb_name_edit and new_kb_name_edit not in get_available_kbs():
                    os.rename(get_kb_path(st.session_state.active_kb), get_kb_path(new_kb_name_edit))
                    db.rename_kb(st.session_state.active_kb, new_kb_name_edit)
                    st.session_state.active_kb = new_kb_name_edit
                    st.session_state.editing_kb_name = False
                    st.rerun()
                else: 
                    st.error("Invalid or existing name.")
        with col2:
            if st.button("Cancel", icon=":material/close:"):
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
        st.session_state.vectorstore = build_knowledge_base(st.session_state.active_kb, load_embeddings_model())
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
