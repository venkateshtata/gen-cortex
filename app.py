from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference.query import semantic_search
from inference.response_generation import generate_response
from config.settings import LLM_MODEL_NAME
from inference.query import get_embedding_model_instance
import sys
from pydantic import BaseModel
import time  # Add this import


app = FastAPI()



class QueryRequest(BaseModel):
    query: str
    num_chunks: int = 5
    stream: bool = False  # Default to False



@app.post("/query")
async def query_endpoint(request: QueryRequest):
    start_time = time.time()  # Start timing
    try:
        # Initialize the embedding model
        embedding_model = get_embedding_model_instance()
        
        # Perform semantic search
        context = semantic_search(request.query, embedding_model, request.num_chunks)
        
        # Generate the response
        response = generate_response(
            llm=LLM_MODEL_NAME,
            query=request.query,
            context=context,
        )
        
        execution_time = time.time() - start_time  # Calculate execution time
        
        return {
            "context": context, 
            "response": response,
            "execution_time_seconds": round(execution_time, 2)  # Round to 2 decimal places
        }
    except Exception as e:
        print(f"Error in /query endpoint: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
