import os
import celery
from flask import Blueprint, jsonify, request
from openai import NotFoundError
from celery.result import AsyncResult
from app.chroma_db import ChromaDB
from app.cluster_classify import ClusterAndClassify
from app.models.doc_file import DocFile
from app.rag import RAG
from app.tasks import process_file_workflow

main = Blueprint('main', __name__)

# Ensure the folder for saving uploaded files exists
UPLOAD_FOLDER = 'cache/uploads'
ML_FOLDER = 'cache/ml-model'


@main.route("/")
def home():
    return "<p>Welcome to EAI!!!</p>"


@main.route("/docs/uploadandtrain", methods=['POST'])
def cluster_and_classify():
    try:
        if 'files[]' not in request.files:
            return jsonify({"error": "No files provided"}), 400
        
        files = request.files.getlist('files[]')
        saved_files = []

        for file in files:
            if file.filename == '':
                return jsonify({"error": "Empty filename"}), 400

            # Save each file to the upload folder
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            print("Filename: ", file.filename)
            saved_files.append(file_path)

        task = process_file_workflow(saved_files)

        return jsonify({"status": "processing", "task_id": task.id}), 202
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@main.route("/docs/classify", methods=['POST'])
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


@main.route('/rag/query', methods=["POST"])
def query_rag():
    try:
        data = request.get_json()
        if data is None:
            raise ValueError("Missing data in the request body")

        rag = RAG(ChromaDB())
        res = rag.process_user_query(data['query'], data["access_level"])

        return jsonify({"response": res[0], "curated_query": res[1], "files": res[2]})
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500
    

@main.route('/rag2/query', methods=["POST"])
def query_rag2():
    try:
        data = request.get_json()
        if data is None:
            raise ValueError("Missing data in the request body")

        rag = RAG(ChromaDB())
        res = rag.process_user_query_screen2(data['query'], data["access_level"], data["user_role"])

        return jsonify({"response": res})
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@main.route('/status/<task_id>')
def get_status(task_id):
    task = AsyncResult(task_id)
    if task.state == 'PENDING':
        return jsonify({"state": task.state})
    elif task.state == 'SUCCESS':
        return jsonify({"state": task.state, "result": task.result})
    else:
        return jsonify({"state": task.state})


@main.route('/docs', methods = ["GET"])
def fetch_files():
    try:
        db = ChromaDB()
        res = db.get()

        return jsonify(res)
    except NotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@main.route('/docs', methods=['PATCH'])
def update_files():
    try:
        data = request.get_json()
        # Check if data is not None (i.e., the JSON body was valid)
        if data is None:
            raise ValueError("Missing data in the request body")
        db = ChromaDB()
        res = db.update_documents(data)
        return jsonify({'status': res})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500
    
@main.route('/docs', methods=['DELETE'])
def delete_files():
    try:
        ChromaDB().delete_all_docs()
        return jsonify({'status': "Deleted all records."})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500
