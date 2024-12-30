from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from config.settings import EMBEDDING_MODEL_NAME

class EmbedChunks:
    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"device": "cuda", "batch_size": 100}
        )

    def __call__(self, batch):
        embeddings = self.embedding_model.embed_documents(batch["text"])
        return {"text": batch["text"], "source": batch["source"], "embeddings": embeddings}
