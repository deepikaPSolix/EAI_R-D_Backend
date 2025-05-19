import os
from vanna.chromadb import ChromaDB_VectorStore
from vanna.openai import OpenAI_Chat

class MyVanna(OpenAI_Chat, ChromaDB_VectorStore):
    def __init__(self,
        config = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": "gpt-4o",
            "path": "../vanna-chroma"
        }):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)