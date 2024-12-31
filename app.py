from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference.query import semantic_search
from inference.response_generation import generate_response
from config.settings import LLM_MODEL_NAME
from inference.query import get_embedding_model_instance
import time
from ray import serve
import torch

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    num_chunks: int = 5
    stream: bool = False

@serve.deployment(
    ray_actor_options={"num_gpus": 1,},
    max_ongoing_requests=1,
    num_replicas=3
)
@serve.ingress(app)  # Ensure the FastAPI app is properly integrated
class QueryService:
    def __init__(self):
        self.embedding_model = get_embedding_model_instance()

        gpu_available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if gpu_available else "No GPU"
        print(f"GPU Available: {gpu_available}, Device Name: {device_name}")
    
    @app.post("/query")
    async def query_endpoint(self, request: QueryRequest):
        start_time = time.time()
        try:
            # semantic_search_start = time.time()
            # context = semantic_search(request.query, self.embedding_model, request.num_chunks)
            # semantic_search_time = time.time() - semantic_search_start

            response_generation_start = time.time()
            response = generate_response(
                llm=LLM_MODEL_NAME,
                query=request.query,
                context=None,
            )
            response_generation_time = time.time() - response_generation_start

            execution_time = time.time() - start_time

            return {
                # "context": context,
                "response": response,
                # "semantic_search_time_seconds": round(semantic_search_time, 2),
                "response_generation_time_seconds": round(response_generation_time, 2),
                "execution_time_seconds": round(execution_time, 2),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
