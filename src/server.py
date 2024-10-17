from dotenv import load_dotenv
from flask import Flask, request, jsonify
import os

from chroma_db import ChromaDB
from cluster_classify import ClusterAndClassify
from attribute_extractor import AttributeExtractor
from doc_file import DocFile
import pandas as pd
from celery import Celery
from celery.result import AsyncResult
import redis

from rag import RAG

app = Flask(__name__)
load_dotenv()


# Configure Celery
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6380/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6380/0'
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'], backend=app.config['CELERY_RESULT_BACKEND'])


# Ensure the folder for saving uploaded files exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Heavy processing task
@celery.task
def process_files(files):
    parsed_files = [DocFile(file_path= path) for path in files] # Extract data from file and put in DocFile Object

    # Cluster Data
    cc = ClusterAndClassify()
    cc_res = cc.cluster_classify(data = parsed_files)
    labels = cc.generate_cluster_labels(cc_res)

    # Extract attributes
    attr_ext = AttributeExtractor()
    attr_res = attr_ext.extract_from_files(parsed_files)
    # labels.to_csv('labels.csv', index=False)
    # attr_res.to_csv('attr.csv', index=False)

    result = pd.merge(labels, attr_res, on='file_name', how='left') 
    result['attributes'] = result[attr_res.columns.difference(['file_name'])].apply(lambda row: row.to_json(), axis=1)
    result = result[['file_name', 'data', 'label','sensitivity', 'attributes']]
    # result.to_csv('result.csv', index=False)

    cdb = ChromaDB()
    cdb.add_documents(result)

    return "Added documents to chromadb!"


@app.route("/")
def home():
    return "<p>Welcome to EAI!!!</p>"


@app.route("/uploadandtrain", methods=['POST'])
def cluster_and_classify():
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    files = request.files.getlist('files')
    saved_files = []

    for file in files:
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        # Save each file to the upload folder
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        print("Filename: ", file.filename)
        saved_files.append(file_path)

    task = process_files.delay(saved_files)

    return jsonify({"status": "processing", "task_id": task.id}), 202

@app.route("/classify", methods=['POST'])
def classify():
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    files = request.files.getlist('files')
    saved_files = []

    for file in files:
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        # Save each file to the upload folder
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        print("Filename: ", file.filename)
        saved_files.append(file_path)

    parsed_files = [DocFile(file_path= path) for path in saved_files] # Extract data from file and put in DocFile Object
    cc = ClusterAndClassify()
    labels = cc.classify_data(parsed_files)

    return jsonify({"labels": labels.to_json(orient='records')}), 201


@app.route('/query_rag')
def query_rag():
    query = request.args.get('query', None)
    access_level = request.args.get('access_level', 1)

    rag = RAG(ChromaDB())
    res = rag.process_user_query(query, access_level)

    return jsonify({"response": res})

@app.route('/status/<task_id>')
def get_status(task_id):
    task = celery.AsyncResult(task_id)
    if task.state == 'PENDING':
        return jsonify({"state": task.state})
    elif task.state == 'SUCCESS':
        return jsonify({"state": task.state, "result": task.result})
    else:
        return jsonify({"state": task.state})

# @app.route("/retrain", methods=['POST'])
# def cluster_and_classify():
#     return "Query result"



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
