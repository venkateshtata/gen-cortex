from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import numpy as np
from rag.generate import prepare_response
from rag.utils import get_client
from .database_utils import DatabaseTool

class QueryAgent:
    def __init__(
        self,
        embed_model_name,
        db_name,
        user,
        passwd,
        host,
        port,
        retrieval_limit,
        llm_name,
        llm_temperature, 
        llm_max_context_length,
        system_content,
        assistant_content,
        stream,
        max_retries,
        retry_interval
    ):
        self.embed_model = HuggingFaceEmbeddings(embed_model_name)
        self.db = DatabaseTool(
            user=user,
            passwd=passwd,
            host=host,
            port=port
        )
        self.retrieval_limit = retrieval_limit
        self.stream = stream
        self.max_retries = max_retries
        self.retry_interval = retry_interval

        llm_max_context_length = int(0.5 * llm_max_context_length)
        self.llm_name = llm_name
        self.llm_temperature = temperature
        self.llm_context_length =  llm_max_context_length - get_num_tokens(system_content + assistant_content)
        self.system_content = system_content
        self.assistant_content = assistant_content


    def _encode_query(self, query):
        print("Encoding query...", end="")
        embedded_query = np.array(self.embed_model.embed_query(query))
        print("Done")

        return embedded_query


    def _generate_response(self, user_content):
        client = get_client(llm=self.llm_name)

        messages = [{"role": role, "content": content} for role, content in [
            ("system", self.system_content), 
            ("assistant", self.assistant_content), 
            ("user", user_content)] if content]

        for _ in range(self.max_retries):
            try:
                chat_completion = client.chat.completions.create(
                    model=self.llm_name,
                    temperature=self.llm_temperature,
                    stream=self.stream,
                    messages=messages,
                )
                return prepare_response(chat_completion, stream=stream)

            except Exception as e:
                print(f"Exception: {e}")
                time.sleep(self.retry_interval)
        return ""

    
    def __call__(self, query):
        embedded_query = self._encode_query(query=query)

        context_results = self.db.retrieve_embeddings(
            query_embedding=embedded_query,
            limit=self.retrieval_limit
        )

        context = [item["text"] for item in context_results]
        sources = [item["source"] for item in context_results]
        user_content = f"query: {query}, context: {context}"

        response = self._generate_response(user_content=trim(user_content, self.llm_context_length))

        result = {
            "question": query,
            "sources": sources,
            "answer": response,
            "llm": self.llm_name
        }

        return result