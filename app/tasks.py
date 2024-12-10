from celery import shared_task
import pandas as pd

from app.attribute_extractor import AttributeExtractor
from app.chroma_db import ChromaDB
from app.cluster_classify import ClusterAndClassify
from app.doc_file import DocFile
from app.utils import UPLOAD_FOLDER, delete_files_in_directory


@shared_task
def process_files(files):
    parsed_files = [DocFile(file_path= path) for path in files] # Extract data from file and put in DocFile Object

    # Cluster Data
    cc = ClusterAndClassify()
    cc_res = cc.cluster_classify(data = parsed_files)
    labels = cc.generate_cluster_labels(cc_res)
    labels.to_csv('cache/labels.csv')
    # Extract attributes
    attr_ext = AttributeExtractor()
    attr_res = attr_ext.extract_from_files(parsed_files)
    attr_res.fillna('')

    result = pd.merge(labels, attr_res, on='file_name', how='left') 
    result['attributes'] = result[attr_res.columns.difference(['file_name'])].apply(lambda row: row.to_dict(), axis=1)
    result = result[['file_name', 'data', 'label','sensitivity', 'data_classifiers', 'responsible_values', 'retention_time', 'attributes']]
    result.to_csv('cache/result.csv')
    cdb = ChromaDB()
    cdb.add_documents(result)
    delete_files_in_directory(UPLOAD_FOLDER)


    return "Added documents to chromadb!"