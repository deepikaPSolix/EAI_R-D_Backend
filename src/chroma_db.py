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
        try:
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
                                'data_classifiers': row['data_classifiers'],
                                'responsible_values': json.dumps(row['responsible_values']),
                                'file_name': doc_name,
                                'retention_time': row['retention_time'],
                                'attributes':json.dumps(doc_attributes),
                            }
                                ]
                        )
            return True
        except Exception as e:
            print("[add_documents] Exception - " + str(e))
            return False

    
    def update_documents(self, data_dict):
        ids = [d['id'] for d in data_dict]
        documents = [d['data'] for d in data_dict]
        metadatas = [{'label':d['label'], 'sensitivity': d['sensitivity'], 'attributes':json.dumps(d['attributes'])} for d in data_dict]
        try:
            res = self.collection.upsert(
                ids = ids,
                documents=documents,
                metadatas = metadatas
            )
            return True
        except Exception as e:
            print("[update_documents] Exception - " + str(e))
            return False

    def get(self, ids = None, where = None):
        try:
            result = self.collection.get(
                ids = ids if ids else None,
                where = where if where else None
            )
            data = []
            n = len(result['ids'])
            for ele in range(n):
                if not result['metadatas'][ele]['label']:
                    print("NO LABEL: " + result['metadatas'][ele]['file_name'])
                data_dict = {
                    'id': result['ids'][ele],
                    'file_name' : result['metadatas'][ele]['file_name'],
                    'data' : result['documents'][ele],
                    'label' : result['metadatas'][ele]['label'],
                    'sensitivity' : result['metadatas'][ele]['sensitivity'],
                    'data_classifiers': result['metadatas'][ele]['data_classifiers'],
                    'responsible_values': result['metadatas'][ele]['responsible_values'],
                    'retention_time' : result['metadatas'][ele]['retention_time'],
                    'attributes' : json.loads(result['metadatas'][ele]['attributes'])
                }
                data.append(data_dict)
            return data
        except Exception as e:
            print("[get] Exception - " + str(e) + str(ele))

    

    def query_db(self, query_text, user_role, k = 20):
        try:
            result = self.collection.query(
                query_texts=[query_text], 
                n_results=k,
                where = {'sensitivity':{'$lte' : int(user_role)}}
                )
            
            data = []
            n = len(result['ids'][0])
            for ele in range(n):
                data_dict = {
                    'file_name' : result['metadatas'][0][ele]['file_name'],
                    'data' : result['documents'][0][ele],
                    'label' : result['metadatas'][0][ele]['label'],
                    'sensitivity' : result['metadatas'][0][ele]['sensitivity'],
                    'attributes' : result['metadatas'][0][ele]['attributes']
                }
                data.append(data_dict)
            return data
        except Exception as e:
            print("[update_documents] Exception - " + str(e))
