import json
import os
from flask import Blueprint, current_app, jsonify, request, send_from_directory, Response
from openai import NotFoundError
from celery.result import AsyncResult
from app.chroma_db import ChromaDB
from app.graph_builder import GraphBuilder
from app.models.doc_file import DocFile
from app.cluster_classify import ClusterAndClassify
from app.rag import RAG
from app.tasks import evaluationFunction, process_file_workflow, screen2EvaluationFunction, parse_graph_file
from app.utils import delete_files
from flask import request, jsonify, send_from_directory
from langchain.vectorstores import FAISS
from typing import Dict
from app.graphFiles import GraphFiles
import asyncio
import re
from app.dashboard import Dashboard
import nest_asyncio

doc_updates = 0

main = Blueprint('main', __name__)

@main.route("/")
def home():
    current_app.logger.info("Welcome to EAI!!!")
    return "<p>Welcome to EAI!!!</p>"


@main.route('/filecontents/<filename>', methods=["GET"])
def serve_file(filename):
    print("Hi")
    print(filename)
    if not filename:
        print("Filename is required")
        return jsonify({"error": "Filename is required"}), 400

    try:
        return send_from_directory(current_app.config['UPLOAD_DIR_PATH'], filename)
    except Exception as e:
        return jsonify({"error": str(e)},exc_info=True), 500


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

        rag = RAG(ChromaDB(),model_source=data['model_name']) 

        res = rag.process_user_query(data['query'], data["access_level"])
       
        combinedList=[res[1],res[0],res[3],data["query"]]
        evaluationFunction.delay(combinedList,include_relevance=True,include_hallucination=True,include_moderation=False,evaluation_result_file="queryEvaluationScreen1Results.json",evaluation_result_csv="queryEvaluationScreen1Results.csv")
        

        return jsonify({"response": res[0], "curated_query": res[1], "files": res[2]})
    except Exception as e:
        current_app.logger.error(str(e), exc_info=True)
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500
    

@main.route('/rag2/query', methods=["POST"])
def query_rag2():
    try:
        data = request.get_json()
        if data is None:
            raise ValueError("Missing data in the request body")

        rag = RAG(ChromaDB(),model_source=data['model_name'])
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

        response_json = {"response": res}

        if(data["lida"] == True):
            dash = Dashboard()
            path=dash.generate_csv_from_response(response=res, model_source="together")
            openai_api_key = os.getenv("OPENAI_API_KEY")
            chart_link=dash.run_lida_on_csv(path, user_query=data["query"]+", represent in "+data["graph_type"] + "chart ", api_key=openai_api_key)

            current_app.logger.info(" Lida : image generated")
            response_json["chart"] = "/"+chart_link
        
        
        return jsonify(response_json)
        
    
    except Exception as e:
        current_app.logger.error(str(e))
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@main.route('/cache/images/<filename>')
def serve_image(filename):
    images_dir = os.path.join('/myapp', 'cache', 'images')  # Absolute path in container

    full_path = os.path.join(images_dir, filename)
    current_app.logger.info(f"Looking for image at: {full_path}")

    if not os.path.exists(full_path):
        current_app.logger.error(f"Image not found: {full_path}")
        return jsonify({"error": "Image not found"}), 404

    return send_from_directory(images_dir, filename)



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
    global doc_updates
    try:
        db = ChromaDB()
        res = db.get()

        if not res:
            res = []

        return jsonify({'documents': res, 'doc_updates': doc_updates})
    except NotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        current_app.logger.error(str(e))
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@main.route('/docs', methods=['PATCH'])
def update_docs():
    global doc_updates
    try:
        data = request.get_json()
        # Check if data is not None (i.e., the JSON body was valid)
        if data is None:
            raise ValueError("Missing data in the request body")
        db = ChromaDB()
        res = db.update_documents(data)
        doc_updates += 1
        return jsonify({'status': res, 'doc_updates': doc_updates})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        current_app.logger.error(str(e))
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500
    
@main.route('/docs', methods=['DELETE'])
def delete_docs():
    global doc_updates
    try:
        ChromaDB().delete_all_docs()
        FILES_TO_CLEAR=["queryEvaluationScreen1Results.json","queryEvaluationScreen2Result.json", "sensitivityEvaluation.json", "fileAttributesResult.json","fileClusterResult.json"]
        file_paths = [os.path.join(current_app.config['BASE_DIR'], file_name) for file_name in FILES_TO_CLEAR]
        delete_files(file_paths)
        doc_updates = 0
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


# data_dir = os.path.join(os.path.dirname(__file__), "data")
# builder = GraphBuilder(data_dir=data_dir)  # Single instance for data consistency
# processor = GraphProcessor(builder)
@main.route("/graph/process", methods=["POST"])
def process_graph():
    nest_asyncio.apply()
    try:
        data = request.get_json()
        if not data or "url" not in data:
            return jsonify({"error": "URL required"}), 400
        builder=GraphBuilder()
        graph=GraphFiles()
        graph.source_url = data["url"] 
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        visited= loop.run_until_complete(
            builder.crawl_website(data["url"], int(data.get("depth",1))))
        chunks = builder.chunk_visited_pages(visited)
        G_nx = graph.build_similarity_graph(chunks)
        html_content = graph.render_graph_html(G_nx, min_cluster_size=7,MAX_LABEL_NODES=23)
        current_app.graph_builder = graph
        return Response(html_content, mimetype="text/html")
    except Exception as e:
        current_app.logger.error(f"Processing error: {str(e)}",exc_info=True)
        return jsonify({"error": str(e)}), 500


@main.route("/graph/uploaddocs", methods=['POST'])
def docupload():
    try:
        if 'files[]' not in request.files:
            return jsonify({"error": "No files provided"}), 400
        
        files = request.files.getlist('files[]')
        current_app.logger.info("FILES ARE BEING PROCESSED..... Hold UP!")
        saved_files = []
        filenames = []
        # Save uploaded files
        for file in files:
            if file.filename == '':
                return jsonify({"error": "Empty filename"}), 400
            
            file_path = os.path.join(current_app.config['GRAPH_DOC_UPLOAD'], file.filename)
            file.save(file_path)
            saved_files.append(file_path)
            filenames.append(file.filename)

        chunks = []
        for file_path in saved_files:
            chunk_list =[chunk.text for chunk in parse_graph_file(file_path)] # 👈 Return only chunks
            chunks.extend(chunk_list)
        current_app.logger.info("CHUNKS CREATED :)")
        graph_builder = GraphFiles()
        graph_builder.file_names = filenames
        G_nx = graph_builder.build_similarity_graph(chunks)
        html_path = graph_builder.render_graph_html(G_nx, min_cluster_size=3,MAX_LABEL_NODES=15)
        current_app.graph_builder = graph_builder
        current_app.logger.error(f"HTML Visualisation Generated", exc_info=True)
        return Response(html_path, mimetype="text/html")
    except Exception as e:
        current_app.logger.error(str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500


@main.route("/graph/query", methods=["POST"])
def graph_query():
    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "Query parameter required"}), 400

        if not hasattr(current_app, "graph_builder"):
            return jsonify({"error": "Graph not ready. Please upload documents or scrape website first."}), 400

        builder = current_app.graph_builder

        result = builder.query_graph_link_response(data["query"])

        # Extract link and clean text from result
        url_pattern = r"https?://\S+"
        links = re.findall(url_pattern, result)
        text_without_links = re.sub(url_pattern, "", result)
        clean_response = re.sub(r"\n+", "\n", text_without_links).strip()

        response_json = {
            "response": clean_response
        }

        # Include the link if it's present
        if links:
            response_json["link"] = links[0]
        current_app.logger.info(f"RESPONSE: {response_json}", exc_info=True)
        return jsonify(response_json)

    except Exception as e:
        current_app.logger.error(f"Graph query error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500
