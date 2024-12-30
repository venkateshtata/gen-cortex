import psycopg
from pgvector.psycopg import register_vector
from config.settings import CONNECTION_STRING

class StoreResults:
    def __call__(self, batch):
        with psycopg.connect(CONNECTION_STRING) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                for text, source, embedding in zip(batch["text"], batch["source"], batch["embeddings"]):
                    cur.execute(
                        "INSERT INTO document (text, source, embedding) VALUES (%s, %s, %s)",
                        (text, source, embedding)
                    )
        return {}
