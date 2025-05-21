import os
from vanna.chromadb import ChromaDB_VectorStore
from vanna.openai import OpenAI_Chat

class MyVanna(OpenAI_Chat, ChromaDB_VectorStore):
    document_store = {}
    def __init__(self,
        config = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": "gpt-4o",
            "path": "../vanna-chroma"
        }):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

    def add_document(self, db_id, doc_id):
        MyVanna.document_store[db_id] = doc_id

    def get_document(self, db_id):
        if db_id in MyVanna.document_store:
            return MyVanna.document_store[db_id]
        else:
            return None
        
    def delete_document(self, db_id):
        if db_id in MyVanna.document_store:
            del MyVanna.document_store[db_id]
        
    def list_documents(self):
        return list(MyVanna.document_store.keys())