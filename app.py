from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference.query import semantic_search
from inference.response_generation import generate_response
from config.settings import EMBEDDING_MODEL_NAME, LLM_MODEL_NAME
from inference.query import get_embedding_model_instance
import sys
from pydantic import BaseModel


app = FastAPI()



class QueryRequest(BaseModel):
    query: str
    num_chunks: int = 5
    stream: bool = False  # Default to False



@app.post("/query")
async def query_endpoint(request: QueryRequest):
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
        
        return {"context": context, "response": response}
    except Exception as e:
        print(f"Error in /query endpoint: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
