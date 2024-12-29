from utils import QueryAgent


def setup_agent(
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
    agent = QueryAgent(
        embed_model_name=embed_model_name,
        db_name=db_name,
        user=user,
        passwd=passwd,
        host=host,
        port=port,
        retrieval_limit=retrieval_limit,
        llm_name=llm_name,
        llm_temperature=llm_temperature, 
        llm_max_context_length=llm_max_context_length,
        system_content=system_content,
        assistant_content=assistant_content,
        stream=stream,
        max_retries=max_retries,
        retry_interval=retry_interval
    )

    return agent;


def get_response(agent, query):
    return agent(query=query)