import os
from pathlib import Path
from celery import shared_task
import pandas as pd

from app.attribute_extractor import AttributeExtractor
from app.chroma_db import ChromaDB
from app.cluster_classify import ClusterAndClassify
from app.models.doc_file import DocFile
from app.dynamic_extractor import DynamicExtractor
from app.file_processor import AudioFileProcessor, GenericFileProcessor, VideoFileProcessor
from app.utils import UPLOAD_FOLDER, delete_files_in_directory


@shared_task
def process_files(files):
    parsed_files = [] # Extract data from file and put in DocFile Object
    generic_processor = GenericFileProcessor()
    audio_processor = None
    video_processor = None
    for f in files:
        file_type = Path(f).suffix[1:]
        file_name = Path(f).name
        if file_type == "mp3":
            if not audio_processor:
                audio_processor = AudioFileProcessor()
            chunks = audio_processor.process_file(f)
        elif file_type == "mp4":
            if not video_processor:
                video_processor = VideoFileProcessor()
            chunks = video_processor.process_file(f)
        else:
            chunks = generic_processor.process_file(f)
        parsed_files.append(DocFile(file_name = file_name, file_type = file_type, chunks = chunks))

    # Cluster Data
    # cc = ClusterAndClassify()
    # cc_res = cc.cluster_classify(data = parsed_files)
    # labels = cc.generate_cluster_labels(cc_res)
    # labels.to_csv('cache/labels.csv')
    # Extract attributes
    attr_ext = DynamicExtractor()
    attr_res = attr_ext.extract_from_files(parsed_files)
    attr_res.fillna('')
    attr_res.to_csv('cache/attrs.csv')

    # result = pd.merge(labels, attr_res, on='file_name', how='left') 
    # result['attributes'] = result[attr_res.columns.difference(['file_name'])].apply(lambda row: row.to_dict(), axis=1)
    # result = result[['file_name', 'data', 'label','sensitivity', 'data_classifiers', 'responsible_values', 'retention_time', 'attributes']]
    # result.to_csv('cache/result.csv')
    # cdb = ChromaDB()
    # cdb.add_documents(result)
    delete_files_in_directory(UPLOAD_FOLDER)


    return "Added documents to chromadb!"