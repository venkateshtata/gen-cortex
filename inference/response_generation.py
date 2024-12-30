from transformers import pipeline
from rag.generate import response_stream

def generate_response(llm, query, context, temperature=0.1):
    """
    Generate a response using the specified LLM.
    """
    generator = pipeline('text-generation', model=llm)
    input_text = f"query: {query}, context: {context}"

    # Generate raw response
    raw_response = generator(
        input_text,
        max_length=4096,
        temperature=temperature,
        do_sample=False  # Greedy decoding
    )

    # Process the generator output
    return response_stream(raw_response[0])  # raw_response is a list, so take the first element
