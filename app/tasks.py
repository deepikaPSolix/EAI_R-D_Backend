from pathlib import Path
from celery import group, shared_task, chain
import pandas as pd

from app.chroma_db import ChromaDB
from app.cluster_classify import Classification, Clustering, LabelGenerator
from app.models.doc_file import DocFile
from app.dynamic_extractor import DynamicExtractor
from app.file_processor import AudioFileProcessor, GenericFileProcessor, VideoFileProcessor
from app.utils import UPLOAD_FOLDER, delete_files_in_directory

def process_file_workflow(files: list):
    job = group([parse_file.s(f) for f in files])
    workflow = chain(
        job, generate_labels.s(),
        insert_to_db.s(),  # Wrap group in a chord and follow with generate_labels
        cleanup.s()  # Add cleanup as the final step in the chain
    )
    result = workflow.apply_async()
    return result

@shared_task()
def process_files(files):
    try:
        pass
        # parsed_files = parse_files(files)
        # parsed_files.to_csv('cache/parsed_files.csv')

        # attr_ext = DynamicExtractor()
        # attr_res = attr_ext.extract(parsed_files[['file_name', 'chunks']])
        # attr_res.to_csv('cache/attrs.csv')

        # parsed_files = pd.merge(parsed_files, attr_res[['file_name', 'attributes']], on='file_name', how='left')


        # # Cluster Data
        # clustering = Clustering()
        # classification = Classification()
        # label_generator = LabelGenerator()

        # partition_index = int(len(parsed_files) * 0.7)
        # clustering_data = parsed_files[:partition_index]
        # classification_data = parsed_files[partition_index:]

        # # Clustering
        # clustering_df = clustering_data[["file_name", "data"]]
        # cluster_labels = clustering.cluster_data(clustering_df)
        # clustering_df["cluster"] = cluster_labels

        # # Train the classifier
        # classification.train(clustering_df["data"], clustering_df["cluster"])

        # # Classify remaining data
        # classification_df = classification_data[["file_name", "data"]]
        # classification_df["cluster"] = classification.classify(
        #     classification_df["data"]
        # )

        # # # Combine results
        # cc_res_df = pd.concat([clustering_df, classification_df], ignore_index=True)
        # parsed_files = pd.merge(parsed_files, cc_res_df[['file_name', 'cluster']], on="file_name", how="left")

        # labeled_data = label_generator.generate_labels(parsed_files[['file_name', 'cluster', 'chunks']])
        # labeled_data.to_csv('cache/labels.csv')
        # parsed_files['cluster_label'] = labeled_data['cluster_label']
        # parsed_files.to_csv('cache/final_df.csv')

    except Exception as e:
        print("Error: " + str(e))
    finally:
        # delete_files_in_directory(UPLOAD_FOLDER)
        pass


    return "Added documents to chromadb!"

@shared_task()
def generate_labels(data: list):
    try:
        df = pd.DataFrame(data)
        clustering = Clustering()
        classification = Classification()
        label_generator = LabelGenerator()

        partition_index = int(len(df) * 0.7)
        clustering_data = df[:partition_index]
        classification_data = df[partition_index:]

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

        cc_res_df = pd.concat([clustering_df, classification_df], ignore_index=True)
        parsed_files = pd.merge(df, cc_res_df[['file_name', 'cluster']], on="file_name", how="left")

        labeled_data = label_generator.generate_labels(parsed_files[['file_name', 'cluster', 'chunks']])
        labeled_data.to_csv('cache/labels.csv')
        parsed_files['cluster_label'] = labeled_data['cluster_label']
        parsed_files.to_csv('cache/final_df.csv')
        return parsed_files.to_dict()
    except Exception as e:
        print("Error: " + str(e))

@shared_task()
def insert_to_db(data):
    try:
        df = pd.DataFrame(data)
        chroma_db = ChromaDB()
        chroma_db.add_documents(df)
    except Exception as e:
        print("Error: " + str(e))

@shared_task()
def parse_file(file_path: str):
    file_type = Path(file_path).suffix[1:]
    file_name = Path(file_path).name
    try:
        if file_type == "mp3":
            audio_processor = AudioFileProcessor()
            chunks = audio_processor.process_file(file_path)
        elif file_type == "mp4":
            video_processor = VideoFileProcessor()
            chunks = video_processor.process_file(file_path)
        else:
            generic_processor = GenericFileProcessor()
            chunks = generic_processor.process_file(file_path)
        attr_ext = DynamicExtractor()
        attr_res = attr_ext.extract_from_file(chunks)
        return {'file_name' : file_name, 'file_type' : file_type, 'chunks': [chunk.text for chunk in chunks], 'data' : " | ".join([chunk.text for chunk in chunks]), 'attributes' : attr_res.model_dump()}
    except Exception as e:
        print("Fn: parse_file Filename: " + file_name + " Error: " + str(e))

@shared_task()
def cleanup(*args):
    try:
        delete_files_in_directory(UPLOAD_FOLDER)
        return "Files deleted!"
    except Exception as e:
        print("Error: " + str(e))
