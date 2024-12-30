
import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from rag.generate import prepare_response
from rag.embed import get_embedding_model
from rag.utils import get_num_tokens, trim
from rag.config import MAX_CONTEXT_LENGTHS
import json
import time
from transformers import pipeline



embedding_model_name = "thenlper/gte-base"
connection_string = f"postgresql://postgres:postgres@localhost:5432/postgres"




def semantic_search(query, embedding_model, k):
    embedding = np.array(embedding_model.embed_query(query))
    with psycopg.connect(connection_string) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM document ORDER BY embedding <=> %s LIMIT %s", (embedding, k),)
            rows = cur.fetchall()
            semantic_context = [{"id": row[0], "text": row[1], "source": row[2]} for row in rows]
    return semantic_context


def generate_response(
    llm, temperature=0.0, stream=True,
    system_content="", assistant_content="", user_content="", 
    max_retries=1, retry_interval=60):
    """Generate response from an LLM using Hugging Face's LLaMA model."""
    retry_count = 0
    
    generator = pipeline('text-generation', model=llm)
    messages = [{"role": role, "content": content} for role, content in [
        ("system", system_content), 
        ("assistant", assistant_content), 
        ("user", user_content)] if content]
    input_text = " ".join([message["content"] for message in messages])
    while retry_count <= max_retries:
        try:
            chat_completion = generator(input_text, max_length=1024, temperature=temperature, stream=stream)
            return prepare_response(chat_completion, stream=stream)

        except Exception as e:
            print(f"Exception: {e}")
            time.sleep(retry_interval)  # default is per-minute rate limits
            retry_count += 1
    return ""


class QueryAgent:
    def __init__(self, embedding_model_name="thenlper/gte-base",
                 llm="meta-llama/Llama-2-7b-chat-hf", temperature=0.0, 
                 max_context_length=4096, system_content="", assistant_content=""):

        # Embedding model
        self.embedding_model = get_embedding_model(
            embedding_model_name=embedding_model_name, 
            model_kwargs={"device": "cuda"}, 
            encode_kwargs={"device": "cuda", "batch_size": 100})

	 # Context length (restrict input length to 50% of total length)
        max_context_length = int(0.5*max_context_length)

        # LLM
        self.llm = llm
        self.temperature = temperature
        self.context_length =  max_context_length - get_num_tokens(system_content + assistant_content)
        self.system_content = system_content
        self.assistant_content = assistant_content

    def __call__(self, query, num_chunks=5, stream=True):
        # Get sources and context
        context_results = semantic_search(
            query=query, 
            embedding_model=self.embedding_model, 
            k=num_chunks)

        # Generate response
        context = [item["text"] for item in context_results]
        sources = [item["source"] for item in context_results]
        user_content = f"query: {query}, context: {context}"

        answer = generate_response(
            llm=self.llm,
            temperature=self.temperature,
            stream=stream,
            system_content=self.system_content,
            assistant_content=self.assistant_content,
            user_content=trim(user_content, self.context_length))

        # Result
        result = {
            "question": query,
            "sources": sources,
            "answer": answer,
            "llm": self.llm,
        }
        return result
    


llm = "meta-llama/Llama-2-7b-chat-hf"
# "meta-llama/Llama-3.2-3B-Instruct"
agent = QueryAgent(
    embedding_model_name="thenlper/gte-base",
    llm=llm,
    max_context_length=MAX_CONTEXT_LENGTHS[llm],
    system_content="Answer the query using the context provided. Be succinct.")
result = agent(query="What is the default batch size for map_batches?")
print("\n\n", json.dumps(result, indent=2))