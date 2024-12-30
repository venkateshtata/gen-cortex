import psycopg
from pgvector.psycopg import register_vector
from config.settings import CONNECTION_STRING

def create_document_table():
    with psycopg.connect(CONNECTION_STRING) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document (
                    id SERIAL PRIMARY KEY,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    embedding vector(768)
                );
            """)
        conn.commit()
