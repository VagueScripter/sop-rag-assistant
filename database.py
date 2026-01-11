import sqlite3
import datetime

DB_NAME = "chats.db"

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def _create_tables(conn):
    """Creates the necessary tables if they don't exist."""
    with conn:
        # Create chat_threads table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_threads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create chat_messages table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (thread_id) REFERENCES chat_threads(id) ON DELETE CASCADE
            )
        """)

        # Create a trigger to update the updated_at timestamp on the threads table
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

def create_new_thread(title: str) -> int:
    """Creates a new chat thread and returns its ID."""
    conn = get_db_connection()
    with conn:
        cursor = conn.execute(
            "INSERT INTO chat_threads (title) VALUES (?)", (title,)
        )
    new_id = cursor.lastrowid
    conn.close()
    return new_id

def get_all_threads():
    """Retrieves all chat threads, sorted by the most recently updated."""
    conn = get_db_connection()
    threads = conn.execute(
        "SELECT * FROM chat_threads ORDER BY updated_at DESC"
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

def add_message_to_thread(thread_id: int, role: str, content: str):
    """Adds a new message to a specific chat thread."""
    conn = get_db_connection()
    with conn:
        conn.execute(
            "INSERT INTO chat_messages (thread_id, role, content) VALUES (?, ?, ?)",
            (thread_id, role, content),
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

def delete_thread(thread_id: int):
    """Deletes a chat thread and all its messages."""
    conn = get_db_connection()
    with conn:
        conn.execute("DELETE FROM chat_threads WHERE id = ?", (thread_id,))
    conn.close()

if __name__ == "__main__":
    # Example usage
    print("Initializing database...")
    init_db()
    print("Database initialized.")

    # Create a new thread
    print("Creating a new chat thread...")
    thread_id = create_new_thread("My First Chat")
    print(f"New thread created with ID: {thread_id}")

    # Add messages
    print("Adding messages...")
    add_message_to_thread(thread_id, "user", "Hello, assistant!")
    add_message_to_thread(thread_id, "assistant", "Hello, user! How can I help you today?")

    # Retrieve messages
    print(f"Retrieving messages for thread {thread_id}...")
    messages = get_messages_by_thread(thread_id)
    for msg in messages:
        print(f"  [{msg['timestamp']}] {msg['role']}: {msg['content']}")

    # List all threads
    print("Listing all threads...")
    all_threads = get_all_threads()
    for thread in all_threads:
        print(f"  ID: {thread['id']}, Title: {thread['title']}, Last Updated: {thread['updated_at']}")

    # Delete the thread
    # print(f"Deleting thread {thread_id}...")
    # delete_thread(thread_id)
    # print("Thread deleted.")
