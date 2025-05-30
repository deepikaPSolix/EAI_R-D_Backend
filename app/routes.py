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
from app.tasks import evaluationFunction, process_file_workflow, screen2EvaluationFunction, parse_graph_file, parse_file_data_only
from app.utils import delete_files
from flask import request, jsonify, send_from_directory
from app.file_processor import AudioFileProcessor
from langchain.vectorstores import FAISS
from typing import Dict
from app.graphFiles import GraphFiles
from app.graph_storage_pg import GraphPostgresStorage
import asyncio
import re
from app.dashboard import Dashboard
import nest_asyncio
import glob
import tempfile
import subprocess
import psycopg2
from docx import Document
from app.vanna_class import MyVanna
doc_updates = 0
main = Blueprint('main', __name__)

@main.route("/")
def home():
    current_app.logger.info("Welcome to EAI!!!")
    return "<p> Welcome to EAI Application solix docker test!!!</p>"


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
@main.route('/transcribe', methods=['POST'])
def transcribe():
    current_app.logger.info(f"Audio received. Processing .......")
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']

    # Save the uploaded WebM blob
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as f:
        audio_file.save(f.name)
        webm_path = f.name

    wav_path = webm_path.replace(".webm", ".wav")

    # Convert to WAV (Whisper prefers it)
    subprocess.run(['ffmpeg', '-i', webm_path, wav_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        processor = AudioFileProcessor()
        transcript = processor.extract_text_from_audio(wav_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(webm_path)
        os.remove(wav_path)
    current_app.logger.info(f"Transcript is , {transcript}")
    return jsonify({'transcript': transcript})

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

@main.route('/vanna/train/doc', methods=["POST"])
def train_vanna_doc():
    try:
        if 'files[]' not in request.files:
            return jsonify({"error": "No files provided"})
        
        files = request.files.getlist('files[]')
        vn = MyVanna()

        for file in files:
            if file.filename == '':
                return jsonify({"error": "Empty filename"})

            # Save each file to the upload folder
            file_path = os.path.join(current_app.config['SQL_UPLOAD_DIR_PATH'], file.filename)
            file.save(file_path)

            data = parse_file_data_only(file_path)
        
            res = vn.train(documentation=data["data"])
            vn.add_document(db_id=res, doc_id=data["file_name"])
            current_app.logger.info(f"✅ Added documentation to vanna chroma:\n{res}\n")

        return jsonify({"ids": [res]}), 202
    except Exception as e:
        current_app.logger.error(str(e))
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500
    
@main.route('/vanna/train/ddl', methods=["POST"])
def train_vanna_ddl():
    try:
        if 'files[]' not in request.files:
            return jsonify({"error": "No files provided"})
        
        files = request.files.getlist('files[]')
        vn = MyVanna()

        for file in files:
            if file.filename == '':
                return jsonify({"error": "Empty filename"})

            # Save each file to the upload folder
            file_path = os.path.join(current_app.config['SQL_UPLOAD_DIR_PATH'], file.filename)
            file.save(file_path)

            data = parse_file_data_only(file_path)
            res = []
            
            full_text = data["data"]
            ddl_statements = re.findall(r'CREATE TABLE.*?\);', full_text, re.DOTALL | re.IGNORECASE)
            
            for ddl in ddl_statements:
                db_id = vn.train(ddl=ddl)
                res.append(db_id)
                vn.add_document(db_id=db_id, doc_id=data["file_name"])

        current_app.logger.info(f"✅ Added {str(len(ddl_statements))} DDL to vanna chroma")
        return jsonify({"ids": res}), 202
    except Exception as e:
        current_app.logger.error(str(e))
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500
    
@main.route('/docs/vanna', methods=["DELETE"])
def del_vanna_training_data():
    vn = MyVanna()
    res = []
    df = vn.get_training_data()
    id_list = df["id"].tolist()
    for id in id_list:
        removed = vn.remove_training_data(id=id)
        res.append({"id": id, "removed" : removed})
    vn.delete_all()
    return res, 202

@main.route('/docs/vanna/id/<id>', methods=["DELETE"])
def del_vanna_training_data_by_id(id):
    vn = MyVanna()
    removed = vn.remove_training_data(id=id)
    vn.delete_document(id)
    return jsonify({"id": id, "removed" : removed}), 202

@main.route('/docs/vanna', methods=["GET"])
def get_vanna_training_data():
    vn = MyVanna()
    df = vn.get_training_data()
    df_list = df.to_dict(orient='list')
    return df_list["id"], 202

@main.route('/docs/vanna/names', methods=["GET"])
def get_vanna_training_data_names():
    vn = MyVanna()
    names = vn.list_document_names()
    return list(names), 202

@main.route('/docs/vanna/id/<id>', methods=["GET"])
def get_vanna_doc_from_db_id(id):
    vn = MyVanna()
    doc_id = vn.get_document(id)
    if doc_id is None:
        return jsonify({"error": "Document not found"}), 404
    return doc_id, 202

@main.route('/rag2/query/vanna', methods=["POST"])
def query_vanna(data=None):
    try:
        if data is None:
            data = request.get_json()
        if data is None:
            raise ValueError("Missing data in the request body")
        vn = MyVanna()

        current_app.logger.info(f"Vanna query: {data['query']}")
        sql, df, _ = vn.ask(
            question=data["query"],
            print_results=False,
            auto_train=True,
            visualize=False,
            allow_llm_to_see_data=False
        )
        
        return jsonify({"response": sql, "query_result": df.to_json(orient='records') if df is not None else None})
    except Exception as e:
        current_app.logger.error(str(e), exc_info=True)
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

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
        res,curated_query,top_reranked_docs = rag.process_user_query_screen2(data['query'], data["access_level"], data["user_role"],  data["lida"])

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

        # Graph -> Remove Python code from the response (if any)
        pattern = r"```python(.*?)```"
        clean_response = re.sub(pattern, "", res, flags=re.DOTALL).strip()

        # Voice feature-- Summary in 50 words
        summary_prompt = (
        f"Summarize the following answer in less than or equal to 50 words.\n"
        f"Curated Query: \"{curated_query}\"\n"
        f"Answer: \"{clean_response}\"")

        summary_resp = rag.model.invoke(summary_prompt)
        summary_text = summary_resp.content.strip()
        # voice feature - end
        
        response_json = {"response": clean_response}
        response_json["summary"] = summary_text
         

        
        if(data["lida"] == True):
            dash = Dashboard()
            path=dash.generate_csv_from_response(response=res, model_source="together")
            openai_api_key = os.getenv("OPENAI_API_KEY")
            chart_link=dash.run_lida_on_csv(path, user_query= curated_query+", represent in "+data["graph_type"] + "chart ", api_key=openai_api_key)
            
            current_app.logger.info(" Lida : image generated")
            response_json["chart"] = "/"+chart_link

        else:
            dash = Dashboard()
            path2 = dash.generate_dashboard(response=res)
            if path2 is not None:
                response_json["chart"] = "/" + path2
            else:
                current_app.logger.warning("⚠️ Graph was requested but no chart was generated.")
                
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
        for csv_file in glob.glob("cache/csv/*.csv"):
            os.remove(csv_file)
            current_app.logger.info(f"Deleted CSV: {csv_file}")

        for python_code in glob.glob("cache/code/*.py"):
            os.remove(python_code)
            current_app.logger.info(f"Deleted Code: {python_code}")

        for img_file in glob.glob("cache/images/*.png"):
            os.remove(img_file)
            current_app.logger.info(f"Deleted image: {img_file}")
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

chunks = None
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
        global chunks
        chunks = builder.chunk_visited_pages(visited)
        graph_id,G_nx = graph.build_similarity_graph(chunks)
        html_content = graph.render_graph_html(G_nx, min_cluster_size=6,MAX_LABEL_NODES=15, threshold=0.98)
        # store_cached_html(key, html_content)
        # store_graph_data(key, graph)  
        GraphPostgresStorage(os.getenv("DSN")).store_graph_metadata(graph_id, source_label=data["url"])
        current_app.graph_builder = graph
        # current_app.graph_cache_key = key
        # GraphSessionHandler.set(graph, key)
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
        global chunks
        chunks = []
        for file_path in saved_files:
            filename = os.path.basename(file_path)
            chunk_list = [f"{filename}||{chunk.text}" for chunk in parse_graph_file(file_path)]  # ⬅️ Add filename inside the chunk with separator
            chunks.extend(chunk_list)
        current_app.logger.info("CHUNKS CREATED :)")
        graph_builder = GraphFiles()
        graph_builder.file_names = filenames
        graph_id,G_nx = graph_builder.build_similarity_graph(chunks)
        html_path = graph_builder.render_graph_html(G_nx, min_cluster_size=1,MAX_LABEL_NODES=15, threshold=0.84)
        
        GraphPostgresStorage(os.getenv("DSN")).store_graph_metadata(graph_id, source_label=", ".join(filenames))
        current_app.graph_builder = graph_builder
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

        if not hasattr(current_app, "graph_builder") or not hasattr(current_app.graph_builder, "all_chunks"):
                # Fallback: load the latest graph from DB
                graph_builder = GraphFiles()
                latest_graph_id = GraphPostgresStorage(os.getenv("DSN")).list_graphs(limit=1)[0]["id"]
                graph_builder.load_graph_from_db(latest_graph_id)
                current_app.graph_builder = graph_builder
        else:
            graph_builder = current_app.graph_builder

        result = graph_builder.query_graph_link_response(data["query"])


        # Extract link and clean text from result
        url_pattern = r"https?://\S+"
        links = re.findall(url_pattern, result["answer"])
        text_without_links = re.sub(url_pattern, "", result["answer"])
        clean_response = re.sub(r"\n+", "\n", text_without_links).strip()

        # Check if the source is a file or a link and format accordingly
        source = result.get("file")
     
        response_json = {
            "response": clean_response,
            "source": source if source else ""
        }

        if clean_response == "**I don't have enough information.**" or clean_response=="I dont't have enough information.":
            response_json = {
                "response": clean_response
            }

        # Include the link if it's present
        if links:
            response_json["link"] = links[0]

        return jsonify(response_json)

    except Exception as e:
        current_app.logger.error(f"Graph query error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

    
@main.route("/graph/render-clusters-merged", methods=["POST"])
def render_semantic_cluster_merge():
    try:
        data = request.get_json()
        current_app.logger.info(f"selected clusters list: {data}", exc_info=True)
        graph = GraphFiles()
        html = graph.render_combined_clusters_to_single_graph(data)

        current_app.graph_builder = graph  # Crucial for /graph/query to work!

        return Response(html, mimetype="text/html")
    except Exception as e:
        current_app.logger.error(f"Error in merged cluster view: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500



def extract_main_label(source_label):
    if not source_label:
        return None
    cleaned = source_label.replace(".pdf", "").strip()

    parts = [part.strip() for part in cleaned.split("-")]

    if len(parts) >= 3:
        return parts[1]
    elif len(parts) == 2:
        return parts[1]
    else:
        return parts[0]
    
def extract_main_labels_from_multiple(source_labels):
    if not source_labels:
        return None

    filenames = [s.strip() for s in source_labels.split(",")]
    main_labels = [extract_main_label(name) for name in filenames]

    return ", ".join(main_labels)



@main.route("/graph/graph-cluster-list", methods=["GET"])
def get_graph_and_clusters():
    try:
        storage = GraphPostgresStorage(os.getenv("DSN"))
        conn = storage.conn
        cur = conn.cursor()

        cur.execute("""
            SELECT
            g.id as graph_id,
            g.source_label,
            g.created_at,  
            cl.cluster_id,
            cl.label as cluster_label,
            COUNT(gn.id) as node_count
        FROM graphs g
        LEFT JOIN cluster_labels cl ON cl.graph_id = g.id
        LEFT JOIN graph_nodes gn ON gn.graph_id = g.id AND gn.cluster_id = cl.cluster_id
        GROUP BY g.id, g.source_label, g.created_at, cl.cluster_id, cl.label
        ORDER BY g.id, cl.cluster_id

        """)

        rows = cur.fetchall()

        graph_map = {}
        for graph_id, source_label, created_at, cluster_id, cluster_label, node_count in rows:
            if cluster_id == -1:
                continue  
            if graph_id not in graph_map:
                graph_map[graph_id] = {
                    "graph_id": graph_id,
                    "source_label": extract_main_labels_from_multiple(source_label) or f"Graph {graph_id}",
                    "created_at": created_at.strftime("%B %d, %Y %I:%M %p"),
                    "clusters": []
                }

            graph_map[graph_id]["clusters"].append({
                "cluster_id": cluster_id,
                "label": cluster_label or f"Cluster {cluster_id}",
                "node_count": node_count
            })

        return jsonify(list(graph_map.values()))

    except Exception as e:
        current_app.logger.error(f"Error listing graph-cluster data: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@main.route("/graph/<int:graph_id>/cluster-labels", methods=["GET"])
def get_cluster_labels(graph_id):
    try:
        conn = psycopg2.connect(os.getenv("DSN"))
        cur = conn.cursor()
        cur.execute("""
            SELECT cluster_id, label FROM cluster_labels WHERE graph_id = %s
        """, (graph_id,))
        labels = [{"cluster_id": row[0], "label": row[1]} for row in cur.fetchall()]
        return jsonify(labels)
    except Exception as e:
        current_app.logger.error(f"Error fetching labels: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500
