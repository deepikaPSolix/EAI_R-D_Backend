import json
import os
from flask import Blueprint, current_app, jsonify, request, Response
from openai import NotFoundError
from celery.result import AsyncResult
from app.chroma_db import ChromaDB
from app.cluster_classify import ClusterAndClassify
from app.models.doc_file import DocFile
from app.rag import RAG
from app.tasks import evaluationFunction, process_file_workflow, screen2EvaluationFunction
from app.utils import delete_files
from flask import request, jsonify, send_from_directory
from app.graph_rag import GraphRAGProcessor
import asyncio
import nest_asyncio

main = Blueprint('main', __name__)

@main.route("/")
def home():
    current_app.logger.info("Welcome to EAI!!!")
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
            file_path = os.path.join(current_app.config['UPLOAD_DIR_PATH'], file.filename)
            file.save(file_path)
            print("Filename: ", file.filename)
            saved_files.append(file_path)

        task = process_file_workflow(saved_files)
        
        return jsonify({"status": "processing", "task_id": task.id}), 202
    except Exception as e:
        current_app.logger.error(str(e))
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
        file_path = os.path.join(current_app.config['UPLOAD_DIR_PATH'], file.filename)
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
       
        combinedList=[res[1],res[0],res[3],data["query"]]
        evaluationFunction.delay(combinedList,include_relevance=True,include_hallucination=True,include_moderation=False,evaluation_result_file="queryEvaluationScreen1Results.json",evaluation_result_csv="queryEvaluationScreen1Results.csv")
        

        return jsonify({"response": res[0], "curated_query": res[1], "files": res[2]})
    except Exception as e:
        current_app.logger.error(str(e))
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500
    

@main.route('/rag2/query', methods=["POST"])
def query_rag2():
    try:
        data = request.get_json()
        if data is None:
            raise ValueError("Missing data in the request body")

        rag = RAG(ChromaDB())
        res,curated_query,top_reranked_docs = rag.process_user_query_screen2(data['query'], data["access_level"], data["user_role"])

        formatted_data = [
            f"filename: {item['file_name']}, text: {item['text']}" for item in top_reranked_docs
        ]

        ## This is for TextAnalysis class (cosine distance)
        combinedList = [ 
            str(res),            
            formatted_data,
            data["query"]    
        ]
        screen2EvaluationFunction.delay(combinedList)
        return jsonify({"response": res})
    except Exception as e:
        current_app.logger.error(str(e))
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
def fetch_docs():
    try:
        db = ChromaDB()
        res = db.get()

        return jsonify(res)
    except NotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        current_app.logger.error(str(e))
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@main.route('/docs', methods=['PATCH'])
def update_docs():
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
        current_app.logger.error(str(e))
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500
    
@main.route('/docs', methods=['DELETE'])
def delete_docs():
    try:
        ChromaDB().delete_all_docs()
        FILES_TO_CLEAR=["queryEvaluationScreen1Results.json","queryEvaluationScreen2Result.json", "sensitivityEvaluation.json", "fileAttributesResult.json","fileClusterResult.json"]
        file_paths = [os.path.join(current_app.config['BASE_DIR'], file_name) for file_name in FILES_TO_CLEAR]
        delete_files(file_paths)
        return jsonify({'status': "success"})
    except Exception as e:
        current_app.logger.error(str(e))
        return jsonify({"status": "failed", "error": f"Internal Server Error: {str(e)}"}), 500
    
# Eval Routes

@main.route('/sensitivityEvalApi',methods=["GET"])
def sensitivityEvalResults():
    try:
        EVALUATION_RESULTS_FILE = './cache/sensitivityEvaluation.json'

        if not os.path.exists(EVALUATION_RESULTS_FILE):
            return jsonify([]), 200

        # Read existing data
        with open(EVALUATION_RESULTS_FILE, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                return jsonify({"error": "Invalid JSON format in evaluation results file."}), 500

        # Return the data as a JSON response
        return jsonify(data), 200

    except Exception as e:
        print(f"Error fetching evaluation results: {e}")
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500
    
@main.route('/attributesEvalApi',methods=["GET"])
def attributesEvalResults():
    try:
        EVALUATION_RESULTS_FILE = './cache/fileAttributesResult.json'

        if not os.path.exists(EVALUATION_RESULTS_FILE):
            return jsonify([]), 200

        # Read existing data
        with open(EVALUATION_RESULTS_FILE, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                return jsonify({"error": "Invalid JSON format in evaluation results file."}), 500

        # Return the data as a JSON response
        return jsonify(data), 200

    except Exception as e:
        print(f"Error fetching evaluation results: {e}")
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500
    
@main.route('/clusterEvalApi',methods=["GET"])
def clusterEvalResults():
    try:
        EVALUATION_RESULTS_FILE = './cache/fileClusterResult.json'

        if not os.path.exists(EVALUATION_RESULTS_FILE):
            return jsonify([]), 200

        # Read existing data
        with open(EVALUATION_RESULTS_FILE, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                return jsonify({"error": "Invalid JSON format in evaluation results file."}), 500

        # Return the data as a JSON response
        return jsonify(data), 200

    except Exception as e:
        print(f"Error fetching evaluation results: {e}")
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500
    
@main.route('/rag1EvalApi',methods=["GET"])
def rag1EvalResults():
    try:
        EVALUATION_RESULTS_FILE = './cache/queryEvaluationScreen1Results.json'

        if not os.path.exists(EVALUATION_RESULTS_FILE):
            return jsonify([]), 200

        # Read existing data
        with open(EVALUATION_RESULTS_FILE, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                return jsonify({"error": "Invalid JSON format in evaluation results file."}), 500

        # Return the data as a JSON response
        return jsonify(data), 200

    except Exception as e:
        print(f"Error fetching evaluation results: {e}")
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500
    

@main.route('/rag2EvalApi',methods=["GET"])
def rag2EvalResults():
    try:
        EVALUATION_RESULTS_FILE = './cache/queryEvaluationScreen2Result.json'

        if not os.path.exists(EVALUATION_RESULTS_FILE):
            return jsonify([]), 200

        # Read existing data
        with open(EVALUATION_RESULTS_FILE, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                return jsonify({"error": "Invalid JSON format in evaluation results file."}), 500

        # Return the data as a JSON response
        return jsonify(data), 200

    except Exception as e:
        print(f"Error fetching evaluation results: {e}")
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500
    
nest_asyncio.apply()
@main.route("/graph/process", methods=["POST"])
def process_graph():
    try:
        data = request.get_json()
        processor = GraphRAGProcessor()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        #level = data.get("depth", 0) 
        graph = loop.run_until_complete(
            processor.build_graph(data["url"], data["domain"])
        )
        
        html_content = processor.generate_visualization(graph)  
        
        processor.create_vector_store(graph)

        return Response(html_content, mimetype="text/html")  
        
    except Exception as e:
        current_app.logger.error(f"Graph processing failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

@main.route("/graph/query", methods=["POST"])
def graph_query():
    try:
        data = request.get_json()
        processor = GraphRAGProcessor()
        processor.load_vector_store()

        results = processor.query_graph(data["query"])
        
        return jsonify({
            "results": results,
            "count": len(results)
        })
    except Exception as e:
        current_app.logger.error(f"Graph query failed: {str(e)}")
        return jsonify({"error": str(e)}), 500