import psycopg2
from pgvector.psycopg import register_vector
import ray

class DatabaseTool:
    def __init__(db_name, user, passwd, host, port):
        print(f"Attempting to connect to {db_name} on {host}...")
        self.conn = psycopg2.connect(
            dbname=db_name,
            user=user,
            password=passwd,
            host=host,
            port=port
        )
        register_vector(self.conn)
        print("Connected")


    @ray.remote
    def store_embeddings(self, batch):
        cursor = self.conn.cursor()

        query = """
        INSERT INTO document (text, source, embeddings)
        VALUES %s
        """

        data = [(doc['text'], doc['source'], doc.get('embeddings')) for doc in batch]
        execute_values(cursor, query, data)
        
        self.conn.commit()
        cursor.close()
        self.conn.close()


    @ray.remote
    def retrieve_embeddings(self, query_embedding, limit):
        print("Obtaining semantic context...", end="")
        with self.conn.cursor() as cur:
                cur.execute("SELECT * FROM document ORDER BY embeddings <=> %s LIMIT %s", (query_embedding, limit),)
                rows = cur.fetchall()
                semantic_context = [{"id": row[0], "text": row[1], "source": row[2]} for row in rows]
        print("Done")
        
        return semantic_context
