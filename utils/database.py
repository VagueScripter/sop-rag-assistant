import sqlite3

DB_NAME = "chats.db"

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def _create_tables(conn):
    """Creates the necessary tables if they don't exist."""
    with conn:
        # Create chat_threads table with a column to link to a knowledge base
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_threads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kb_name TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create parsed_documents table to store extracted PDF contents and metadata
        conn.execute("""
            CREATE TABLE IF NOT EXISTS parsed_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kb_name TEXT NOT NULL,
                file_name TEXT NOT NULL,
                title TEXT,
                sop_number TEXT,
                content TEXT NOT NULL,
                page_number INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create chat_messages table (no changes needed here)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                sources TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (thread_id) REFERENCES chat_threads(id) ON DELETE CASCADE
            )
        """)

        # Trigger is still valid, no changes needed
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS update_chat_threads_updated_at
            AFTER INSERT ON chat_messages
            FOR EACH ROW
            BEGIN
                UPDATE chat_threads
                SET updated_at = CURRENT_TIMESTAMP
                WHERE id = NEW.thread_id;
            END;
        """)

def init_db():
    """Initializes the database and creates tables."""
    conn = get_db_connection()
    _create_tables(conn)
    conn.close()

# --- PARSED DOCUMENTS ---

def save_parsed_document(kb_name: str, file_name: str, content: str, title: str = None, sop_number: str = None, page_number: int = None):
    """Saves a parsed document chunk to the database."""
    conn = get_db_connection()
    with conn:
        conn.execute(
            "INSERT INTO parsed_documents (kb_name, file_name, title, sop_number, content, page_number) VALUES (?, ?, ?, ?, ?, ?)",
            (kb_name, file_name, title, sop_number, content, page_number)
        )
    conn.close()

def get_parsed_documents(kb_name: str, file_name: str = None):
    """Retrieves all parsed document chunks for a given KB and optional filename."""
    conn = get_db_connection()
    query = "SELECT * FROM parsed_documents WHERE kb_name = ?"
    params = [kb_name]
    if file_name:
        query += " AND file_name = ?"
        params.append(file_name)
    docs = conn.execute(query, params).fetchall()
    conn.close()
    return docs

def delete_parsed_documents(kb_name: str, file_name: str = None):
    """Deletes parsed documents for a KB (and optionally a specific file)."""
    conn = get_db_connection()
    with conn:
        if file_name:
            conn.execute("DELETE FROM parsed_documents WHERE kb_name = ? AND file_name = ?", (kb_name, file_name))
        else:
            conn.execute("DELETE FROM parsed_documents WHERE kb_name = ?", (kb_name,))
    conn.close()

# --- CHAT THREADS ---

def create_new_thread(title: str, kb_name: str) -> int:
    """Creates a new chat thread associated with a knowledge base and returns its ID."""
    conn = get_db_connection()
    with conn:
        cursor = conn.execute(
            "INSERT INTO chat_threads (title, kb_name) VALUES (?, ?)", (title, kb_name)
        )
    new_id = cursor.lastrowid
    conn.close()
    return new_id

def get_all_threads(kb_name: str):
    """Retrieves all chat threads for a specific knowledge base, sorted by the most recently updated."""
    conn = get_db_connection()
    threads = conn.execute(
        "SELECT * FROM chat_threads WHERE kb_name = ? ORDER BY updated_at DESC", (kb_name,)
    ).fetchall()
    conn.close()
    return threads

def get_messages_by_thread(thread_id: int):
    """Retrieves all messages for a given thread ID."""
    conn = get_db_connection()
    messages = conn.execute(
        "SELECT * FROM chat_messages WHERE thread_id = ? ORDER BY timestamp ASC",
        (thread_id,),
    ).fetchall()
    conn.close()
    return messages

def add_message_to_thread(thread_id: int, role: str, content: str, sources: str = None):
    """Adds a new message to a specific chat thread, optionally including sources."""
    conn = get_db_connection()
    with conn:
        conn.execute(
            "INSERT INTO chat_messages (thread_id, role, content, sources) VALUES (?, ?, ?, ?)",
            (thread_id, role, content, sources),
        )
    conn.close()

def get_thread_title(thread_id: int) -> str:
    """Gets the title of a specific thread."""
    conn = get_db_connection()
    title = conn.execute(
        "SELECT title FROM chat_threads WHERE id = ?", (thread_id,)
    ).fetchone()
    conn.close()
    return title['title'] if title else "Chat"

def rename_kb(old_kb_name: str, new_kb_name: str):
    """Renames a knowledge base in the database."""
    conn = get_db_connection()
    with conn:
        conn.execute(
            "UPDATE chat_threads SET kb_name = ? WHERE kb_name = ?",
            (new_kb_name, old_kb_name)
        )
    conn.close()

def delete_kb_threads(kb_name: str):
    """Deletes all chat threads associated with a specific knowledge base."""
    conn = get_db_connection()
    with conn:
        # First, get all thread IDs for the given kb_name
        thread_ids_cursor = conn.execute(
            "SELECT id FROM chat_threads WHERE kb_name = ?", (kb_name,)
        )
        thread_ids = [row[0] for row in thread_ids_cursor.fetchall()]
        
        if thread_ids:
            # Delete messages for those threads
            conn.execute(
                f"DELETE FROM chat_messages WHERE thread_id IN ({','.join('?' for _ in thread_ids)})",
                thread_ids
            )
            # Delete the threads themselves
            conn.execute(
                "DELETE FROM chat_threads WHERE kb_name = ?", (kb_name,)
            )
        
        # Also delete parsed documents when KB is deleted
        conn.execute("DELETE FROM parsed_documents WHERE kb_name = ?", (kb_name,))
    conn.close()

def delete_thread(thread_id: int):
    """Deletes a single chat thread and all its messages."""
    conn = get_db_connection()
    with conn:
        conn.execute("DELETE FROM chat_threads WHERE id = ?", (thread_id,))
    conn.close()

def rename_thread(thread_id: int, new_title: str):
    """Renames a specific chat thread."""
    conn = get_db_connection()
    with conn:
        conn.execute("UPDATE chat_threads SET title = ? WHERE id = ?", (new_title, thread_id))
    conn.close()

def delete_messages_from(message_id: int):
    """Deletes a message and all subsequent messages in the same thread."""
    conn = get_db_connection()
    with conn:
        msg = conn.execute("SELECT thread_id, timestamp FROM chat_messages WHERE id = ?", (message_id,)).fetchone()
        if msg:
            conn.execute("DELETE FROM chat_messages WHERE thread_id = ? AND timestamp >= ?", (msg['thread_id'], msg['timestamp']))
    conn.close()