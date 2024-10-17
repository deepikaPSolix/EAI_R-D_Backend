import chromadb
import uuid
import json

class ChromaDB:

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            # If no instance exists, create one
            cls._instance = super(ChromaDB, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path="./chromadb")
        self.collection = self.create_collection()

    def create_collection(self, name = 'documents'):
        return self.chroma_client.get_or_create_collection(name = name)


    def add_documents(self, data_df):
        for _,row in data_df.iterrows():
            doc_name = row['file_name']
            doc_data = row['data']
            doc_label = row['label']
            doc_sensitivity = row['sensitivity']
            doc_attributes = row['attributes']
        
            self.collection.upsert(
                    documents= [doc_data],
                    ids= [str(uuid.uuid4())],
                    metadatas= [
                        {
                            'label':doc_label,
                            'sensitivity':doc_sensitivity,
                            'attributes':doc_attributes,
                            'file_name': doc_name
                        }
                            ]
                    )
        print('Documents added succesfully')

    
    def update_documents(self, data_dict):
        for ele in data_dict:
            doc_id = ele['id']
            doc_label = ele['label']
            doc_sensitivity = ele['sensitivity']
            doc_attributes = ele['attributes']
        
            self.collection.update(
                    ids= [doc_id],
                    metadatas= [
                        {
                            'label':doc_label,
                            'sensitivity':doc_sensitivity,
                            'attributes':str(doc_attributes)
                        }
                            ]
                    )
        print('Documents updated succesfully')

    def get(self, ids = None, where = None):
        result = self.collection.get(
            ids = ids if ids else None,
            where = where if where else None
        )
        data = []
        n = len(result['ids'])
        for ele in range(n):
            data_dict = {
                'id': result['ids'][ele],
                'file_name' : result['metadatas'][ele]['file_name'],
                'data' : result['documents'][ele],
                'label' : result['metadatas'][ele]['label'],
                'sensitivity' : result['metadatas'][ele]['sensitivity'],
                'attributes' : json.loads(result['metadatas'][ele]['attributes'])
            }
            data.append(data_dict)
        return data
    

    def query_db(self, query_text, user_role, k = 20):
        result = self.collection.query(
            query_texts=[query_text], 
            n_results=k,
            # where = {'sensitivity':{'$lte' : int(user_role)}}
            )
        
        data = []
        n = len(result['ids'][0])
        for ele in range(n):
            # print(ele)
            data_dict = {
                'file_name' : result['metadatas'][0][ele]['file_name'],
                'data' : result['documents'][0][ele],
                'label' : result['metadatas'][0][ele]['label'],
                'sensitivity' : result['metadatas'][0][ele]['sensitivity'],
                'attributes' : result['metadatas'][0][ele]['attributes']
            }
            data.append(data_dict)
        return data
