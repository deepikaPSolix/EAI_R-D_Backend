import chromadb
import uuid
import json

from flask import current_app
from pandas import DataFrame

#Postgres
from app.postgres_db import DatabaseManager
import os
import datetime

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

                metadata = {
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
                
                # Add Excel RCC data if available
                if 'rcc_result' in row and row['rcc_result']:
                    metadata['has_rcc_matches'] = True
                    metadata['rcc_result'] = json.dumps(row['rcc_result'])
                    metadata['matched_tables_count'] = len(row['rcc_result'])
                    metadata['total_tables_count'] = len(row['rcc_result'])  # For now, assume all tables are processed
                else:
                    metadata['has_rcc_matches'] = False
                    metadata['rcc_result'] = json.dumps([])
                
                # Keep backward compatibility with excel_rcc_data if it exists
                if 'excel_rcc_data' in row and row['excel_rcc_data'] and isinstance(row['excel_rcc_data'], dict):
                    metadata['has_rcc_matches'] = row['excel_rcc_data'].get('has_rcc_matches', False)
                    metadata['excel_rcc_assignments'] = json.dumps(row['excel_rcc_data'].get('table_rcc_assignments', []))
                    metadata['matched_tables_count'] = row['excel_rcc_data'].get('matched_tables', 0)
                    metadata['total_tables_count'] = row['excel_rcc_data'].get('total_tables', 0)
                
                metadatas.append(metadata)
            
            self.collection.add(documents= documents, ids= ids, metadatas= metadatas)
            #Postgres DB
            for file_id, data in zip(ids, metadatas):                
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

                try:
                    db_manager=DatabaseManager()
                    db_manager.add_file_metadata(file_metadata_result)
                    db_manager.add_file_metadata_classification(file_metadata_classification_result)
                    current_app.logger.info(f"✅ Inserted metadata for file: {data['file_name']}")
                except Exception as e:
                    current_app.logger.error(f"❌ Error inserting metadata for file '{data['file_name']}': {str(e)}", exc_info=True)            
            
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
                
                # Add Excel RCC data if available
                if result['metadatas'][ele].get('has_rcc_matches'):
                    data_dict['has_rcc_matches'] = True
                    data_dict['rcc_result'] = json.loads(result['metadatas'][ele].get('rcc_result', '[]'))
                    data_dict['matched_tables_count'] = result['metadatas'][ele].get('matched_tables_count', 0)
                    data_dict['total_tables_count'] = result['metadatas'][ele].get('total_tables_count', 0)
                    
                    # Keep backward compatibility
                    if 'excel_rcc_assignments' in result['metadatas'][ele]:
                        data_dict['excel_rcc_assignments'] = json.loads(result['metadatas'][ele].get('excel_rcc_assignments', '[]'))
                
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
            # db_manager=DatabaseManager()
            # db_manager.delete_all_rows()
        except Exception as e:
            current_app.logger.error(str(e))
            print("[delete_all_docs] Exception - " + str(e))    
    def list_collections(self):
        """List all collections in the ChromaDB instance"""
        try:
            # In v0.6.0+, this returns just the collection names (strings)
            return self.chroma_client.list_collections()
        except Exception as e:
            current_app.logger.error(f"Error listing collections: {e}")
            return []
    def cleanup_session_collections(self, max_collections=10, preserve_newest=5):
        """
        Clean up old session-specific collections to prevent unlimited growth.
        
        Args:
            max_collections: Maximum number of collections to keep
            preserve_newest: Number of newest collections to always preserve
            
        Returns:
            Number of collections deleted
        """
        try:
            # Get all collections (in v0.6.0+ this returns just the collection names)
            all_collection_names = self.list_collections()
            if not all_collection_names:
                current_app.logger.warning("No collections found during cleanup")
                return 0
                
            # Filter for session-specific collections - ensure we're working with strings
            session_collection_names = []
            for name in all_collection_names:
                # Handle both string names and collection objects for backwards compatibility
                if hasattr(name, 'name'):
                    collection_name = name.name  # Pre-v0.6.0
                else:
                    collection_name = str(name)  # v0.6.0+
                    
                if collection_name.startswith("graph_chunks_"):
                    session_collection_names.append(collection_name)
            
            # If we're under the limit, no need to delete
            if len(session_collection_names) <= max_collections:
                current_app.logger.info(f"Only {len(session_collection_names)} collections found, under limit of {max_collections}")
                return 0
                
            # Sort collections by name (as a proxy for creation time)
            # Extract session ID and use it for sorting (newer sessions have higher IDs)
            def get_session_id(name):
                try:
                    # Try to extract the session ID (usually the part after the last underscore)
                    return name.split('_')[-1]
                except:
                    return name
                    
            session_collection_names.sort(key=get_session_id)
            
            # Keep the newest collections
            names_to_delete = session_collection_names[:-preserve_newest]
            
            # Only delete if we're over the maximum
            if len(session_collection_names) - len(names_to_delete) < max_collections:
                names_to_delete = session_collection_names[:-(max_collections)]
            
            current_app.logger.info(f"Found {len(session_collection_names)} collections, deleting {len(names_to_delete)}")
            
            # Delete old collections
            deleted_count = 0
            for collection_name in names_to_delete:
                try:
                    self.chroma_client.delete_collection(name=collection_name)
                    current_app.logger.info(f"🧹 Deleted old ChromaDB collection: {collection_name}")
                    deleted_count += 1
                except Exception as e:
                    current_app.logger.warning(f"⚠️ Failed to delete ChromaDB collection {collection_name}: {e}")
            
            return deleted_count
        except Exception as e:
            current_app.logger.error(f"❌ Error during ChromaDB cleanup: {e}")
            return 0