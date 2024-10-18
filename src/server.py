from dotenv import load_dotenv
from flask import Flask, request, jsonify
import os
from chroma_db import ChromaDB
from cluster_classify import ClusterAndClassify
from attribute_extractor import AttributeExtractor
from doc_file import DocFile
import pandas as pd
from celery import Celery
from chromadb.errors import NotFoundError
from rag import RAG

app = Flask(__name__)
load_dotenv()


# Configure Celery
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'], backend=app.config['CELERY_RESULT_BACKEND'])


# Ensure the folder for saving uploaded files exists
UPLOAD_FOLDER = 'uploads'
ML_FOLDER = 'ml-model'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(ML_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def delete_files_in_directory(directory_path):
    try:
        # Loop through all items in the directory
        for item in os.listdir(directory_path):
            # Create full path
            item_path = os.path.join(directory_path, item)
            
            # Check if it's a file (not a directory)
            if os.path.isfile(item_path):
                os.remove(item_path)
                print(f"Deleted: {item_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


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

    result = pd.merge(labels, attr_res, on='file_name', how='left') 
    result['attributes'] = result[attr_res.columns.difference(['file_name'])].apply(lambda row: row.to_json(), axis=1)
    result = result[['file_name', 'data', 'label','sensitivity', 'attributes']]

    cdb = ChromaDB()
    res = cdb.add_documents(result)
    if res:
        delete_files_in_directory(UPLOAD_FOLDER)

    return "Added documents to chromadb!"


@app.route("/")
def home():
    return "<p>Welcome to EAI!!!</p>"


@app.route("/docs/uploadandtrain", methods=['POST'])
def cluster_and_classify():
    try:
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
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@app.route("/docs/classify", methods=['POST'])
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


@app.route('/rag/query', methods=["POST"])
def query_rag():
    try:
        data = request.get_json()
        if data is None:
            raise ValueError("Missing data in the request body")

        rag = RAG(ChromaDB())
        res = rag.process_user_query(data['query'], data["access_level"])

        return jsonify({"response": res[0], "curated_query": res[1]})
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@app.route('/status/<task_id>')
def get_status(task_id):
    task = celery.AsyncResult(task_id)
    if task.state == 'PENDING':
        return jsonify({"state": task.state})
    elif task.state == 'SUCCESS':
        return jsonify({"state": task.state, "result": task.result})
    else:
        return jsonify({"state": task.state})


@app.route('/docs', methods = ["GET"])
def fetch_files():
    try:
        db = ChromaDB()
        res = db.get()

        if not res:
            raise NotFoundError("Could not fetch documents.")

        return jsonify(res)
    except NotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@app.route('/docs', methods=['PATCH'])
def update_files():
    try:
        data = request.get_json()

        # Check if data is not None (i.e., the JSON body was valid)
        if data is None:
            raise ValueError("Missing data in the request body")
        db = ChromaDB()
        db.update_documents([data])
        return jsonify({'status': "success"})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
