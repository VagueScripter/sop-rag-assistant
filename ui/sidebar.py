import streamlit as st
from utils import database as db
from services.kb import get_available_kbs

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        load_css("style.css")
        st.title("SOP Bot")

        available_kbs = get_available_kbs()
        if not available_kbs:
            st.warning("No knowledge bases available.")
            return

        selected_kb = st.selectbox("Select Knowledge Base", available_kbs, key="kb_selector", index=available_kbs.index(st.session_state.active_kb) if st.session_state.active_kb in available_kbs else 0)
        
        if selected_kb and selected_kb != st.session_state.active_kb:
            st.session_state.update({"active_kb": selected_kb, "vectorstore": None, "active_thread_id": None, "view": "chat"})
            st.rerun()
        
        st.divider()
        
        if st.session_state.active_kb:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("New Chat", icon=":material/add:", use_container_width=True):
                    st.session_state.active_thread_id = None
                    st.session_state.view = "chat"
                    st.rerun()
            with col2:
                if st.button("Settings", icon=":material/settings:", use_container_width=True):
                    st.session_state.view = "settings"
                    st.rerun()
            
            st.divider()
            st.header("Chat History")
            threads = db.get_all_threads(st.session_state.active_kb)
            for thread in threads:
                if st.session_state.get(f"renaming_thread_{thread['id']}"):
                    new_title = st.text_input("New Title", value=thread['title'], key=f"rename_input_{thread['id']}")
                    col_save, col_cancel = st.columns(2)
                    with col_save:
                        if st.button("Save", key=f"save_btn_{thread['id']}"):
                            db.rename_thread(thread['id'], new_title)
                            st.session_state[f"renaming_thread_{thread['id']}"] = False
                            st.rerun()
                    with col_cancel:
                        if st.button("Cancel", key=f"cancel_btn_{thread['id']}"):
                            st.session_state[f"renaming_thread_{thread['id']}"] = False
                            st.rerun()
                else:
                    col_thread, col_menu = st.columns([0.85, 0.15])
                    with col_thread:
                        if st.button(thread['title'], key=f"thread_btn_{thread['id']}", use_container_width=True):
                            if st.session_state.active_thread_id != thread['id']:
                                st.session_state.active_thread_id = thread['id']
                                st.session_state.view = "chat"
                                st.rerun()
                    with col_menu:
                        with st.popover("⋮", use_container_width=True):
                            if st.button("Rename", icon=":material/edit:", key=f"edit_thread_{thread['id']}", use_container_width=True):
                                st.session_state[f"renaming_thread_{thread['id']}"] = True
                                st.rerun()
                            if st.button("Delete", icon=":material/delete:", key=f"del_thread_{thread['id']}", use_container_width=True):
                                db.delete_thread(thread['id'])
                                if st.session_state.active_thread_id == thread['id']:
                                    st.session_state.active_thread_id = None
                                st.rerun()
