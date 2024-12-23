import os
from pathlib import Path
from celery import shared_task
import pandas as pd

from app.attribute_extractor import AttributeExtractor
from app.chroma_db import ChromaDB
from app.cluster_classify import Classification, ClusterAndClassify, Clustering, LabelGenerator
from app.models.doc_file import DocFile
from app.dynamic_extractor import DynamicExtractor
from app.file_processor import AudioFileProcessor, FileProcessor, GenericFileProcessor, VideoFileProcessor
from app.utils import UPLOAD_FOLDER, delete_files_in_directory


@shared_task
def process_files(files):
    try:
        parsed_files = parse_files(files)
        # parsed_files.to_csv('cache/parsed_files.csv')

        attr_ext = DynamicExtractor()
        attr_res = attr_ext.extract(parsed_files[['file_name', 'chunks']])
        attr_res.to_csv('cache/attrs.csv')

        parsed_files = pd.merge(parsed_files, attr_res[['file_name', 'attributes']], on='file_name', how='left')


        # Cluster Data
        clustering = Clustering()
        classification = Classification()
        label_generator = LabelGenerator()

        partition_index = int(len(parsed_files) * 0.7)
        clustering_data = parsed_files[:partition_index]
        classification_data = parsed_files[partition_index:]

        # Clustering
        clustering_df = clustering_data[["file_name", "data"]]
        cluster_labels = clustering.cluster_data(clustering_df)
        clustering_df["cluster"] = cluster_labels

        # Train the classifier
        classification.train(clustering_df["data"], clustering_df["cluster"])

        # Classify remaining data
        classification_df = classification_data[["file_name", "data"]]
        classification_df["cluster"] = classification.classify(
            classification_df["data"]
        )

        # # Combine results
        cc_res_df = pd.concat([clustering_df, classification_df], ignore_index=True)
        parsed_files = pd.merge(parsed_files, cc_res_df[['file_name', 'cluster']], on="file_name", how="left")

        labeled_data = label_generator.generate_labels(parsed_files[['file_name', 'cluster', 'chunks']])
        labeled_data.to_csv('cache/labels.csv')
        parsed_files['cluster_label'] = labeled_data['cluster_label']
        parsed_files.to_csv('cache/final_df.csv')

    except Exception as e:
        print("Error: " + str(e))
    finally:
        delete_files_in_directory(UPLOAD_FOLDER)


    return "Added documents to chromadb!"

@shared_task(bind=True)
def parse_files(files: list):
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


        parsed_files.append({'file_name' : file_name, 'file_type' : file_type, 'chunks' : chunks, 'data' : " | ".join([chunk.text for chunk in chunks])})

    return pd.DataFrame(parsed_files)

@shared_task(bind=True, max_retries=3)
def parse_file(file_path: str, processor: FileProcessor):
    return processor.process_file(file_path)
