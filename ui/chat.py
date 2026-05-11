import os
import json
import streamlit as st
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
import streamlit.components.v1 as components

from utils import database as db
from services.kb import get_kb_index_path, load_knowledge_base
from services.llm import load_llm, load_embeddings_model, get_qa_prompt, get_contextualize_prompt, generate_thread_title


def render_inline_editor(msg_id: int, current_content: str):
    """Renders a ChatGPT-style inline editable bubble using custom HTML/JS + a hidden Streamlit form for submission."""
    escaped = current_content.replace("`", "\\`").replace("\n", "\\n")

    components.html(f"""
    <style>
        body {{ margin: 0; padding: 0; background: transparent; font-family: 'Söhne', ui-sans-serif, -apple-system, BlinkMacSystemFont, sans-serif; }}
        .edit-wrap {{
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            width: 100%;
        }}
        textarea#edit-area {{
            width: 100%;
            min-height: 80px;
            background-color: #2f2f2f;
            color: #ececec;
            border: 1px solid #555;
            border-radius: 18px;
            padding: 14px 18px;
            font-size: 15px;
            line-height: 1.6;
            resize: vertical;
            outline: none;
            box-sizing: border-box;
            font-family: inherit;
        }}
        textarea#edit-area:focus {{
            border-color: #888;
        }}
        .edit-actions {{
            display: flex;
            gap: 8px;
            margin-top: 10px;
            align-items: center;
        }}
        .btn-cancel {{
            background: transparent;
            border: none;
            color: #aaa;
            font-size: 14px;
            cursor: pointer;
            padding: 6px 12px;
            border-radius: 8px;
        }}
        .btn-cancel:hover {{ color: #fff; background: rgba(255,255,255,0.08); }}
        .btn-send {{
            background: #fff;
            border: none;
            color: #000;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            padding: 6px 14px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .btn-send:hover {{ background: #e0e0e0; }}
    </style>
    <div class="edit-wrap">
        <textarea id="edit-area">{current_content}</textarea>
        <div class="edit-actions">
            <button class="btn-cancel" onclick="cancelEdit()">Cancel</button>
            <button class="btn-send" onclick="submitEdit()">
                <svg xmlns="http://www.w3.org/2000/svg" height="16" viewBox="0 -960 960 960" width="16" fill="currentColor">
                    <path d="M120-160v-240l320-80-320-80v-240l760 320-760 320Z"/>
                </svg>
                Send
            </button>
        </div>
    </div>
    <script>
        const ta = document.getElementById('edit-area');

        function submitEdit() {{
            const val = ta.value;
            // Write value into Streamlit's query param mechanism via postMessage
            window.parent.postMessage({{
                type: 'streamlit:setComponentValue',
                value: JSON.stringify({{ action: 'submit', msg_id: {msg_id}, content: val }})
            }}, '*');
        }}
        function cancelEdit() {{
            window.parent.postMessage({{
                type: 'streamlit:setComponentValue',
                value: JSON.stringify({{ action: 'cancel', msg_id: {msg_id}, content: '' }})
            }}, '*');
        }}
    </script>
    """, height=180, scrolling=False)


def render_chat_page():
    if st.session_state.active_kb and st.session_state.get("vectorstore") is None:
        index_path = get_kb_index_path(st.session_state.active_kb)
        if os.path.exists(index_path) and os.listdir(index_path):
             with st.spinner(f"Loading '{st.session_state.active_kb}'..."):
                embeddings = load_embeddings_model()
                st.session_state.vectorstore = load_knowledge_base(st.session_state.active_kb, embeddings)
        else:
            st.warning(f"KB '{st.session_state.active_kb}' is empty. Go to Settings to upload and build.")

    is_editing = any(
        st.session_state.get(f"editing_msg_{k}") 
        for k in [m['id'] for m in (db.get_messages_by_thread(st.session_state.active_thread_id) if st.session_state.active_thread_id else [])]
    )

    if st.session_state.active_thread_id:
        chat_history = db.get_messages_by_thread(st.session_state.active_thread_id)
        for i, msg in enumerate(chat_history):
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.markdown("<div class='user-msg-marker'></div>", unsafe_allow_html=True)

                    if st.session_state.get(f"editing_msg_{msg['id']}"):
                        # Store edit value in session state via a hidden text_input (no label shown)
                        edit_key = f"edit_val_{msg['id']}"
                        if edit_key not in st.session_state:
                            st.session_state[edit_key] = msg["content"]

                        # Render custom inline editor
                        new_content = st.text_area(
                            label="",
                            value=st.session_state[edit_key],
                            key=f"textarea_{msg['id']}",
                            label_visibility="collapsed",
                        )

                        col_cancel, col_send = st.columns([1, 1])
                        with col_cancel:
                            if st.button("Cancel", key=f"cancel_{msg['id']}", use_container_width=True):
                                st.session_state[f"editing_msg_{msg['id']}"] = False
                                st.session_state.pop(edit_key, None)
                                st.rerun()
                        with col_send:
                            if st.button("Send", icon=":material/send:", key=f"send_{msg['id']}", use_container_width=True, type="primary"):
                                db.delete_messages_from(msg['id'])
                                st.session_state[f"editing_msg_{msg['id']}"] = False
                                st.session_state.pop(edit_key, None)
                                st.session_state.auto_submit_query = new_content
                                st.rerun()
                    else:
                        st.markdown(msg["content"])
                        if not is_editing:
                            if st.button("", icon=":material/edit:", key=f"edit_btn_{msg['id']}", help="Edit message"):
                                st.session_state[f"editing_msg_{msg['id']}"] = True
                                st.rerun()
                else:
                    st.markdown("<div class='assistant-msg-marker'></div>", unsafe_allow_html=True)
                    st.markdown(msg["content"])
                    if msg["sources"]:
                        st.caption(f"Sources: {json.loads(msg['sources'])}")

                    if i >= len(chat_history) - 2 and not is_editing:
                        if st.button("", icon=":material/refresh:", key=f"regen_btn_{msg['id']}", help="Regenerate"):
                            db.delete_messages_from(msg['id'])
                            last_user_msg = next((m for m in reversed(chat_history) if m["role"] == "user"), None)
                            if last_user_msg:
                                st.session_state.regenerate_query = last_user_msg["content"]
                            st.rerun()
                        if st.button("", icon=":material/content_copy:", key=f"copy_btn_{msg['id']}", help="Copy"):
                            safe_text = msg["content"].replace('`', '\\`').replace('\n', '\\n')
                            js = f"<script>navigator.clipboard.writeText(`{safe_text}`);</script>"
                            components.html(js, height=0, width=0)
                            st.toast("Copied!", icon="✅")
    else:
        st.markdown(f"<p style='color:#888; margin-top: 2rem; text-align:center;'>Start a conversation with <b>{st.session_state.active_kb}</b></p>", unsafe_allow_html=True)

    if not is_editing:
        user_query = st.chat_input("Message...")

        if st.session_state.get("auto_submit_query"):
            user_query = st.session_state.auto_submit_query
            st.session_state.auto_submit_query = None

        is_regenerating = False
        if st.session_state.get("regenerate_query"):
            user_query = st.session_state.regenerate_query
            st.session_state.regenerate_query = None
            is_regenerating = True

        if user_query:
            if st.session_state.vectorstore is None:
                st.error("Knowledge base not loaded. Please build it in Settings.")
            else:
                llm = load_llm()
                if st.session_state.active_thread_id is None:
                    new_title = generate_thread_title(llm, user_query)
                    st.session_state.active_thread_id = db.create_new_thread(new_title, st.session_state.active_kb)

                if not is_regenerating:
                    with st.chat_message("user"):
                        st.markdown("<div class='user-msg-marker'></div>", unsafe_allow_html=True)
                        st.markdown(user_query)
                    db.add_message_to_thread(st.session_state.active_thread_id, "user", user_query)

                with st.chat_message("assistant"):
                    with st.spinner(""):
                        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
                        h_a_retriever = create_history_aware_retriever(llm, retriever, get_contextualize_prompt())
                        qa_chain = create_stuff_documents_chain(llm, get_qa_prompt())
                        rag_chain = create_retrieval_chain(h_a_retriever, qa_chain)
                        history = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in db.get_messages_by_thread(st.session_state.active_thread_id)]

                        try:
                            response = rag_chain.invoke({"input": user_query, "chat_history": history})

                            if "I don't have an answer" in response["answer"]:
                                sources = []
                            else:
                                sources = list(set(os.path.basename(doc.metadata.get("source", "")) for doc in response.get("context", [])))

                            st.markdown(response["answer"])
                            if sources:
                                st.caption(f"Sources: {sources}")

                            db.add_message_to_thread(st.session_state.active_thread_id, "assistant", response["answer"], sources=json.dumps(sources) if sources else None)
                        except Exception as e:
                            st.error(f"Error: {e}")
                            db.add_message_to_thread(st.session_state.active_thread_id, "assistant", f"*(Error: {e})*")
