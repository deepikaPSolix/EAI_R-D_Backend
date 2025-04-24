import chromadb
import uuid
import json

from flask import current_app
from pandas import DataFrame

#Postgres
from app.postgres_db import DatabaseManager
import os

class ChromaDB:

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            # If no instance exists, create one
            cls._instance = super(ChromaDB, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path="./cache/chromadb")
        self.collection = self.create_collection()

    def create_collection(self, name = 'documents'):
        return self.chroma_client.get_or_create_collection(name = name)


    def add_documents(self, data_df: DataFrame):
        try:
            ids = [str(uuid.uuid4()) for _ in range(len(data_df))]
            documents = data_df['data'].tolist()
            metadatas = []
            for _,row in data_df.iterrows():
                doc_name = row['file_name']
                doc_label = row['cluster_label']
                doc_sensitivity = int(row['attributes']['sensitivity'])
                doc_attributes = row['attributes']['attributes']
                doc_attributes["file_type"] = row['file_type']

                metadatas.append(
                     {
                        'label':doc_label,
                        'sensitivity':doc_sensitivity,
                        'data_classifiers': ", ".join(row['attributes']['data_classifiers']),
                        'responsible_values': ", ".join(row['attributes']['responsible_values']),
                        'file_name': doc_name,
                        'retention_time': row['attributes']['retention_time'],
                        'attributes':json.dumps(doc_attributes),
                        'file_size': row['file_size'],
                        'created_at': row['created_at'],
                        'word_count': row['word_count']
                    }
                )
            
            self.collection.add(documents= documents, ids= ids, metadatas= metadatas)
                        #Postgres DB
            current_app.logger.info(f"Documents : {documents}, ID: {ids}, Metadata : {metadatas}")            
            current_app.logger.info(f"Processing {type(ids)} files")
            current_app.logger.info(f"Processing {type(metadatas)} files")
            current_app.logger.info(f"Documents : {documents}, ID: {ids}, Metadata : {metadatas}")


            for file_id, data in zip(ids, metadatas):
                current_app.logger.info(f"looping")
                current_app.logger.info(f"looping {file_id}, {data}")
                current_app.logger.info("ID: %s", file_id)                
                current_app.logger.info("Label: %s", data["label"])
                current_app.logger.info("Sensitivity: %s", data["sensitivity"])
                current_app.logger.info("Data Classifiers: %s", data["data_classifiers"])
                current_app.logger.info("Responsible Values: %s", data["responsible_values"])
                current_app.logger.info("File Name: %s", data["file_name"])
                current_app.logger.info("Retention Time: %s", (lambda s: 0 if not s.strip() else float(s.split()[0]) + (float(s.split(',')[1].split()[0]) / 12 if ',' in s else 0))(data["retention_time"]))
                current_app.logger.info("Attributes: %s", data["attributes"])
                current_app.logger.info("File Size: %s", data["file_size"])
                current_app.logger.info("Created At: %s", data["created_at"])
                current_app.logger.info("Word Count: %s", data["word_count"])
                current_app.logger.info("-----")
                # map numeric codes → descriptive strings
                SENSITIVITY_LABELS = {
                    1: "Public Data",
                    2: "Internal Data",
                    3: "Confidential Data",
                    4: "Restricted Data",
                    5: "Private Data",
                    6: "Critical Data",
                    7: "Regulatory Data",
                }

                file_metadata_result = {    'file_id': file_id,
                                            'file_name': data["file_name"],
                                            'file_type':  os.path.splitext(data["file_name"])[1].lstrip('.') ,
                                            'created_time': data["created_at"],
                                            'last_modification_time': data["created_at"],
                                            'created_by': "System",
                                            'modified_by': "System",
                                            'file_size': data["file_size"]
                                            }

                file_metadata_classification_result = {'file_id': file_id,
                                                       'file_name': data["file_name"],
                                                       'word_count': data["word_count"],
                                                       'data_category': data["label"],
                                                       'sensitivity': SENSITIVITY_LABELS.get(int(data["sensitivity"]), data["sensitivity"]),
                                                       'data_classifiers': data["data_classifiers"],
                                                       'responsible_values': data["responsible_values"], 
                                                       'retention_time': (lambda s: 0 if not s.strip() else float(s.split()[0]) + (float(s.split(',')[1].split()[0]) / 12 if ',' in s else 0))(data["retention_time"]),
                                                       'word_count' : data["word_count"],
                                                       'attributes': data["attributes"],
                                                       'created_by': "System",
                                                       'modified_by': "System",
                                                       }
                current_app.logger.info(f" hit")
                current_app.logger.info(f"File metadata data: {file_metadata_result}")
                current_app.logger.info(f"File metadata data classification: {file_metadata_classification_result}")

                try:
                    db_manager=DatabaseManager()
                    current_app.logger.info(f" db_manager:{ db_manager}")
                    db_manager.add_file_metadata(file_metadata_result)
                    db_manager.add_file_metadata_classification(file_metadata_classification_result)                    
                except Exception as e:
                    current_app.logger.error(f"Error in add_file_metadata: {e}")
            # db_manager.close()
            return True
        except Exception as e:
            print("[add_documents] Exception - " + str(e))
            return False

    
    def update_documents(self, data_dict):

        ids = [d["id"] for d in data_dict]
        documents = [d["data"] for d in data_dict]

        metadatas = []
        for d in data_dict:
            attrs = d.get("attributes", {})
     
            if "reason_for_change" in d:
                attrs["reason_for_change"] = d["reason_for_change"]
            
            metadatas.append({
                "label": d.get("label", ""),
                "sensitivity": int(d.get("sensitivity", 1)),
                "data_classifiers": d.get("data_classifiers", ""),
                "responsible_values": d.get("responsible_values", ""),
                "file_name": d.get("file_name", ""),
                "retention_time": d.get("retention_time", ""),
                "attributes": json.dumps(d.get("attributes", {})),
            })

        try:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
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
                data_dict = {
                    'id': result['ids'][ele],
                    'file_name' : result['metadatas'][ele]['file_name'],
                    'data' : result['documents'][ele],
                    'label' : result['metadatas'][ele].get('label', "No Label"),
                    'sensitivity' : result['metadatas'][ele].get('sensitivity', 0),
                    'data_classifiers': result['metadatas'][ele].get('data_classifiers', ""),
                    'responsible_values': result['metadatas'][ele].get('responsible_values', 'None'),
                    'retention_time' : result['metadatas'][ele].get('retention_time', "None"),
                    'file_size': result['metadatas'][ele].get('file_size', 0),
                    'created_at': result['metadatas'][ele].get('created_at', "None"),
                    'attributes' : json.loads(result['metadatas'][ele]['attributes']),
                    'word_count': result['metadatas'][ele].get('word_count', 0)
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
                    'data_classifiers' : result['metadatas'][0][ele]['data_classifiers'],
                    'responsible_values' : result['metadatas'][0][ele]['responsible_values'],
                    'retention_time' : result['metadatas'][0][ele]['retention_time'],
                    'file_size': result['metadatas'][0][ele].get('file_size', 0),
                    'created_at': result['metadatas'][0][ele].get('created_at', "None"),
                    'attributes' : result['metadatas'][0][ele]['attributes'],
                    "word_count": result['metadatas'][0][ele].get('word_count', 0)
                    
                }
                data.append(data_dict)
            return data
        except Exception as e:
            current_app.logger.error(str(e), exc_info=True)

    def delete_all_docs(self):
        try:
            collection_name = self.collection.name
            self.chroma_client.delete_collection(name=collection_name)
            self.collection = self.chroma_client.create_collection(name=collection_name)
            #Postgres
            db_manager=DatabaseManager()
            db_manager.delete_all_rows()
            # db_manager.close()
        except Exception as e:
            current_app.logger.error(str(e))
            print("[delete_all_docs] Exception - " + str(e))
