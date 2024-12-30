import numpy as np
from rag.embed import get_embedding_model
from config.settings import CONNECTION_STRING, EMBEDDING_MODEL_NAME
import psycopg
from pgvector.psycopg import register_vector

def get_embedding_model_instance():
    return get_embedding_model(
        embedding_model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"device": "cuda", "batch_size": 100}
    )

def semantic_search(query, embedding_model, k):
    embedding = np.array(embedding_model.embed_query(query))
    with psycopg.connect(CONNECTION_STRING) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM document ORDER BY embedding <=> %s LIMIT %s",
                (embedding, k)
            )
            rows = cur.fetchall()
    return [{"id": row[0], "text": row[1], "source": row[2]} for row in rows]
