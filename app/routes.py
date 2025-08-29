import json
import os

from flask import Blueprint, current_app, jsonify, request, send_from_directory, url_for, Response
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
# import uvloop
# uvloop.install()
from app.test import get_trials_sync
from app.llm_model import LLMModel
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
import pandas as pd
import vanna

from app.hpostgres import store_in_postgres

from app.postgres_db import DatabaseManager
import os

import glob
import tempfile
import subprocess
import psycopg2
from docx import Document
from app.vanna_class import MyVanna
from app.test import get_trials_sync 
from app.llm_model import LLMModel
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from app.db_utils import open_db_connection        


doc_updates = 0
main = Blueprint('main', __name__)
dns_host = os.getenv("DNS_HOST")
dns_dbname = os.getenv("DNS_DBNAME")
dns_user = os.getenv("DNS_USER")
dns_password = os.getenv("DNS_PASSWORD")
dns_port = os.getenv("DNS_PORT")
dns = f"host={dns_host} dbname={dns_dbname} user={dns_user} password={dns_password} port={dns_port}"

@main.route("/")
def home():
    current_app.logger.info("Welcome to EAI!!!")
    return "<p> Welcome to EAI Application solix docker test!!!</p>"


@main.route("/trials", methods=["GET"])
def trials_and_analyze_inline():
    # 1) grab query params
    condition = request.args.get("condition","")
    if not condition:
        return jsonify({"error": "Missing required parameter: condition"}), 400
    phase = request.args.get("phase")
    interventions   = request.args.get("interventions")         
    status          = request.args.get("status")                
    study_type      = request.args.get("study_type")
    nct_ids         = request.args.get("nct_ids")  
    size  = request.args.get("size", default=10, type=int)

    try:
        # 2) fetch your trials
        trials = get_trials_sync(condition, phase, interventions, status, study_type, nct_ids,size)
        # serialize for the prompt
        trials_json = json.dumps(trials, ensure_ascii=False)

        # 3) spin up an OpenAI chat model
        chat = ChatOpenAI(
            temperature=0.1,
            model="gpt-4o-mini",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        # 4) build messages: system + user
        messages = [
            SystemMessage(content="You are a clinical-trials research assistant."),
            HumanMessage(
                content=(
                    "Here is a JSON array of trial records:\n\n"
                    f"{trials_json}\n\n"
                    "Please output EXACTLY a JSON object with two keys:\n"
                    "  1) \"summary\": less than or equal to 10-sentence overview of what these trials study\n"
                    "  2) \"top_ids\": an array of the 3 NCT IDs you judge most promising\n\n"
                    "Do NOT output any extra text or markdown."
                )
            )
        ]

        # 5) call the model
        result = chat(messages)
        text = result.content.strip()

        # 6) parse the model’s JSON (fall back to raw text on failure)
        try:
            analysis = json.loads(text)
        except json.JSONDecodeError:
            current_app.logger.error("Failed to parse JSON from LLM:", exc_info=True)
            analysis = {"error_parsing_llm_output": text}

    except Exception as e:
        current_app.logger.error(f"Error in trial lookup or LLM call: {e}", exc_info=True)
        return jsonify({"error": "Internal error fetching or analyzing trials"}), 500

    # 7) return both raw and analyzed
    return jsonify({
        "trials": trials,
        "analysis": analysis
    }), 200


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
        files = request.files.getlist('files[]') if 'files[]' in request.files else []
        
        vn = MyVanna()

        files = request.files.getlist('files[]')
        #  If no valid doc uploaded, call fallback
        if not files or all(f.filename.strip() == '' for f in files):
            db_id = vn.train_with_fallback_doc()
            return jsonify({"id": db_id, "status": "trained with generated metadata"}), 202

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
    
def create_ddl_from_csv(filename, df, vn):
    # Use file name as tabel name
    table_name = filename

    # Find data types
    sys_prompt =  (
        """ You are a senior data analyst. Given the column names and a small sample of table data, analyze the structure and actual data patterns to infer relationships, constraints, and meaning.
            For each column, output the column in quotes followed by the infered data type and parameters as they would be written in a postgresql CREATE TABLE SQL statement. Make sure they are comma separated.
            Infer logical joins, even if not defined in the DDL."""
    )

    messages = [
        vn.system_message(sys_prompt),
        vn.user_message(df.head(10).to_json())
    ]
    print(f"Context Sending to LLM: {messages}")
    generated_doc = vn.submit_prompt(messages)
    sql = vn.extract_sql(generated_doc)

    # Build the DDL statement
    ddl = f"CREATE TABLE {table_name} (\n    {sql}\n);"
    current_app.logger.info(f"Generated DDL from {filename} CSV:\n {ddl}")
    return ddl

def create_metadata_from_csv_ddl(ddl_statements, df, vn):
    context_md = ddl_statements + "\n" + df.head(10).to_json()
    sys_prompt =  (
        """ You are a senior data  analyst. Given the SQL DDL and a small sample of table data, analyze the structure and actual data patterns to infer relationships, constraints, and meaning.
            For each table, output metadata in Markdown starting with: Table: <table_name>
            Then generate a 5-column table with headers:
            | Column Name | Data Type | Constraints | Default/Foreign Key | Description |
            Identify inferred primary keys, foreign keys, and unique columns based on value patterns.
            Leave “—” for blanks.
            Give clear descriptions of what each column represents based on data content.
            Infer logical joins, even if not defined in the DDL.
            Do not output SQL or narrative—only the structured metadata per table."""
    )

    messages = [
        vn.system_message(sys_prompt),
        vn.user_message(context_md)
    ]
    print(f"Context Sending to LLM: {messages}")
    
    generated_doc = vn.submit_prompt(messages)

    return generated_doc

@main.route('/vanna/train/rawdata', methods=["POST"])
def train_vanna_raw():
    try:
        if 'files[]' not in request.files:
            return jsonify({"error": "No files provided"})
        
        files = request.files.getlist('files[]')
        vn = MyVanna()
        res = []
        

        for file in files:
            if file.filename == '':
                return jsonify({"error": "Empty filename"})

            # Save each file to the upload folder
            file_path = os.path.join(current_app.config['SQL_UPLOAD_DIR_PATH'], file.filename)
            file.save(file_path)

            # Get DDLs and Metadata
            try:
                df = pd.read_excel(file_path)
            except ValueError as e:
                try:
                    df = pd.read_csv(file_path)
                except ValueError as e:
                    raise ValueError
                
            ddls = create_ddl_from_csv(file.filename, df, vn)
            metadata_doc = create_metadata_from_csv_ddl(ddls, df, vn)

            ddl_statements = re.findall(r'CREATE TABLE.*?\);', ddls, re.DOTALL | re.IGNORECASE)
            
            for ddl in ddl_statements:
                db_id = vn.train(ddl=ddl)
                res.append(db_id)
                vn.add_document(db_id=db_id, doc_id=file.filename)
            current_app.logger.info(f"✅ Added {str(len(ddl_statements))} DDL to vanna chroma")

            doc_res = vn.train(documentation=metadata_doc)
            res.append(doc_res)
            vn.add_document(db_id=doc_res, doc_id=file.filename)
            current_app.logger.info(f"✅ Added documentation to vanna chroma:\n{doc_res}\n")

        return jsonify({"ids": res}), 202
    except Exception as e:
        current_app.logger.error(str(e))
        return jsonify({"error": f"Internal Server Error at train_vanna_raw(): {str(e)}"}), 500

@main.route('/dbconnect', methods=["POST"])
def store_db_details(details: Dict[str, str]):
    """
    Store database connection details in env.
    """
    os.environ["DB_HOST"] = details["DBHostName"]
    os.environ["DB_PORT"] = details["DBPort"]
    os.environ["DB_NAME"] = details["DBName"]
    os.environ["DB_USER"] = details["DBUserName"]
    os.environ["DB_PASSWORD"] = details["DBPassword"]
    
    return jsonify({"message": "Database connection details stored successfully"}), 200


@main.route('/dbconnect', methods=["GET"])
def get_db_connection(details: Dict[str, str]):
    
    conn = open_db_connection()
    current_app.logger.info("Connected!")
    return conn, 200

@main.route('/dbconnect', methods=["DELETE"])
def delete_db_details():
    """
    Delete database connection details in env.
    """
    if os.getenv("DB_HOST"): del os.environ["DB_HOST"]
    if os.getenv("DB_PORT"): del os.environ["DB_PORT"]
    if os.getenv("DB_NAME"): del os.environ["DB_NAME"]
    if os.getenv("DB_USER"): del os.environ["DB_USER"]
    if os.getenv("DB_PASSWORD"): del os.environ["DB_PASSWORD"]
    
    return jsonify({"message": "Database connection details deleted successfully"}), 200

@main.route('/dbconnect/details', methods=["GET"])
def get_db_connection_details():
    return jsonify({"host":os.getenv("DB_HOST"), "dbname":os.getenv("DB_NAME"), "user":os.getenv("DB_USER")}), 200
    
@main.route('/vanna/dbconnect', methods=["POST"])
def vanna_db_connect():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Extract connection details from the request
    db_host = data.get("DBHostName")
    db_port = data.get("DBPort")
    db_name = data.get("DBName")
    db_user = data.get("DBUserName")
    db_password = data.get("DBPassword")

    # Establish a database connection
    current_app.logger.info(f"Attempting DB connection to {db_host}:{db_port}/{db_name} as {db_user}")

    try:
        vn = MyVanna()
        vn.connect_to_postgres(
            host=db_host,
            dbname=db_name,
            user=db_user,
            password=db_password,
            port=db_port
        )
    except Exception as e:
        current_app.logger.error(f"DB connection failed: {str(e)}")
        return jsonify({"success": False, "message": f"Failed to connect to database: {str(e)}"})

    # A successful connection
    current_app.logger.info(f"DB connection successful")
    store_db_details({
        "DBHostName": db_host,
        "DBPort": db_port,
        "DBName": db_name,
        "DBUserName": db_user,
        "DBPassword": db_password
    })
    return jsonify({"success": True, "message": "Connected to database successfully"}), 200

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
    result_reason = f"No results found for query"
    try:
        if data is None:
            data = request.get_json()
        if data is None:
            raise ValueError("Missing data in the request body")
        vn = MyVanna()
        try:
            vn.connect_to_postgres(
                host= os.getenv("DB_HOST"),
                dbname= os.getenv("DB_NAME"),
                user= os.getenv("DB_USER"),
                password= os.getenv("DB_PASSWORD"),
                port= os.getenv("DB_PORT")
            )
        except Exception as e:
            current_app.logger.error(f"Vanna connection failed")
            result_reason = f"Connect to database to run query"
        
        vanna_query = data['query']
        current_app.logger.info(f"Vanna query: {vanna_query}")
        sql, df, fig = vn.ask(
                question=vanna_query,
                print_results=False,
                auto_train=False,
                visualize=True,
                allow_llm_to_see_data=True
            )
        
        counter = 0
        while (type(df) == Exception or type(df) == vanna.exceptions.ValidationError) and counter < 2:
            current_app.logger.info("Vanna.AI run_sql error occurred:", df)
            current_app.logger.info(f"Vanna.AI run_sql query attempt {counter + 1}:")
            # ✅ Ask Vanna.AI a question
            sql, df, fig = vn.ask(
                question=vanna_query,
                print_results=False,
                auto_train=False,
                visualize=True,
                allow_llm_to_see_data=True
            )

            if type(df) == Exception or type(df) == vanna.exceptions.ValidationError:
                vanna_query += f"\n Attempted query: {sql} Result: {df}"
                counter += 1
        
        current_app.logger.info(f"SQL Query: {sql}")

        if type(df) == Exception or type(df) == vanna.exceptions.ValidationError: # error
            df_result = None
            result_reason = f"SQL error: {df}"
        elif df is None: # no data result
            current_app.logger.info("DataFrame is None, no data returned from Vanna.")
            df_result = None
        else: # success
            current_app.logger.info(type(df))
            current_app.logger.info(f"DataFrame shape: {df.shape}")
            df_result = df.head().to_json(orient='split')
            result_reason = None

        return jsonify({"response": sql, "query_result": df_result, "result_reason": result_reason, "fig": fig.to_json() if fig is not None else None}), 200
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
       
        combinedList=[res[1],res[0],res[3],data["query"], data['model_name'], data['access_level']]
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
            data["query"],
            data['model_name'],
             data["user_role"]  
        ]
        screen2EvaluationFunction.delay(combinedList)

        # Graph -> Remove Python code from the response (if any)
        pattern = r"```python(.*?)```"
        clean_response = re.sub(pattern, "", res, flags=re.DOTALL).strip()

        # Voice feature-- Summary in 50 words
        summary_prompt = (
        f"Summarize the following answer in less than or equal to 30 words.\n"
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
        store_in_postgres(data)
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
        db_manager=DatabaseManager()
        # db_manager.delete_all_rows()
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
        current_app.logger.info(f"CHUNKS from Webpages : {chunks[0]}")
        graph_id,G_nx = graph.build_similarity_graph(chunks)
        html_content = graph.render_graph_html(G_nx, min_cluster_size=6,MAX_LABEL_NODES=15, threshold=0.80)
        # store_cached_html(key, html_content)
        # store_graph_data(key, graph)  
        GraphPostgresStorage(dns).store_graph_metadata(graph_id, source_label=data["url"])
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
        for file in files:
            if file.filename == '':
                return jsonify({"error": "Empty filename"}), 400
            file_path = os.path.join(current_app.config['GRAPH_DOC_UPLOAD'], file.filename)
            # Preserve the original filename exactly as uploaded
            original_filename = file.filename
            file.save(file_path)
            current_app.logger.info(f"[UPLOAD] Saved file: '{original_filename}' at '{file_path}'")
            saved_files.append(file_path)
            filenames.append(original_filename)
        global chunks
        chunks = []        
        for file_path in saved_files:
            # Get the exact original filename (not altered by os.path.basename)
            file_index = saved_files.index(file_path)
            if file_index < len(filenames):
                filename = filenames[file_index]  # Use the original filename
            else:
                filename = os.path.basename(file_path)
            
            current_app.logger.info(f"[CHUNK] Using filename for chunking: '{filename}'")
            chunk_list = [f"{filename}||{chunk.text}" for chunk in parse_graph_file(file_path)]
            chunks.extend(chunk_list)
        current_app.logger.info("CHUNKS CREATED :)")
        # current_app.logger.info(f"CHUNKS from FileUploads : {chunks[0]}")
        graph_builder = GraphFiles()
        graph_builder.file_names = filenames
        graph_id,G_nx = graph_builder.build_similarity_graph(chunks)
        html_path = graph_builder.render_graph_html(G_nx, min_cluster_size=1,MAX_LABEL_NODES=15, threshold=0.70)
        
        GraphPostgresStorage(dns).store_graph_metadata(graph_id, source_label=", ".join(filenames))
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
            return jsonify({"error": "Graph not ready. Please upload first."}), 400

        graph_builder = current_app.graph_builder
        if not hasattr(graph_builder, "all_chunks"):
            # fallback: load *from RedisGraph* if in‐memory chunks aren't set
            graph_builder.load_graph_from_redis()
            current_app.graph_builder = graph_builder

        # Get the response from GraphFiles
        result = graph_builder.query_graph_link_response(data["query"])
        clean_text = result.get('answer', '').strip()
        sources = result.get('sources', [])

        # Build the response JSON
        response_json = {"response": clean_text}

        # Format sources (top 3)
        formatted_sources = []
        for src in sources[:3]:
            if isinstance(src, dict):
                # Already formatted as {name, url}
                formatted_sources.append(src)
            elif isinstance(src, str):
                url_pattern = r"^https?://"
                if re.match(url_pattern, src):
                    formatted_sources.append({"name": src, "url": src})                
                else:                    
                    # Don't use os.path.basename which might normalize spaces
                    # Extract filename from the path while preserving all spaces
                    if '||' in src:
                        # Handle the case where source might be in format "filename||content"
                        filename = src.split('||')[0]
                    else:
                        # For regular paths, extract last part
                        filename = src.split('/')[-1].split('\\')[-1]
                    try:
                        # Check if file exists in the upload directory to ensure correct spaces
                        upload_dir = current_app.config['GRAPH_DOC_UPLOAD']
                        exact_filename = None
                        
                        # Look for the exact filename on disk
                        for existing_file in os.listdir(upload_dir):
                            # Compare filenames ignoring spaces to find the match
                            if existing_file.replace(" ", "") == filename.replace(" ", ""):
                                exact_filename = existing_file
                                current_app.logger.info(f"Found exact filename: '{exact_filename}' for '{filename}'")
                                break
                        
                        # Use the exact filename from disk with correct spacing
                        filename_to_use = exact_filename if exact_filename else filename
                        
                        # Generate URL with the exact filename from disk
                        file_url = url_for('main.download_graph_file', filename=filename_to_use, _external=True)
                        formatted_sources.append({"name": filename_to_use, "url": file_url})
                    except Exception as e:
                        current_app.logger.warning(f"⚠️ Skipped building file URL due to: {e}")
        if formatted_sources:
            response_json["sources"] = formatted_sources

        # Add a summary for voice features
        rag = RAG(ChromaDB(), model_source="together")
        summary_prompt = (
            f"Summarize the following answer in less than or equal to 30 words.\n"
            f"Curated Query: \"{data['query']}\"\n"
            f"Answer: \"{clean_text}\"")
        summary_resp = rag.model.invoke(summary_prompt)
        summary_text = summary_resp.content.strip()
        response_json["summary"] = summary_text
        filename = result.get("sources","")
        if links:
            response_json["link"] = links[0]
        else:
            filename = result.get("sources", "")
            if filename:
                try:
                    file_url = url_for('main.download_graph_file', filename=filename, _external=True)
                    response_json["sources"] = [{"name": filename, "url": file_url}]
                except Exception as e:
                    current_app.logger.warning(f"⚠️ Skipped building file URL due to: {e}")

        return jsonify(response_json)

    except Exception as e:
        current_app.logger.error(f"Graph query error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@main.route('/graph/download/<path:filename>')
def download_graph_file(filename):
    current_app.logger.info(f"[DOWNLOAD] Requested filename: '{filename}'")
    
    # URL decode the filename to preserve spaces and special characters
    import urllib.parse
    decoded_filename = urllib.parse.unquote(filename)
    current_app.logger.info(f"[DOWNLOAD] Decoded filename: '{decoded_filename}'")
    
    # Find the file in the directory - extra safety to ensure exact match
    upload_dir = current_app.config['GRAPH_DOC_UPLOAD']
    file_path = os.path.join(upload_dir, decoded_filename)
    
    # If file doesn't exist with exact name, try more flexible matching
    if not os.path.exists(file_path):
        current_app.logger.warning(f"[DOWNLOAD] File not found with exact match: '{file_path}'")
        found = False
        
        # Check files in the directory, ignoring spaces
        for existing_file in os.listdir(upload_dir):
            # First try exact name without spaces
            if existing_file.replace(" ", "") == decoded_filename.replace(" ", ""):
                decoded_filename = existing_file
                current_app.logger.info(f"[DOWNLOAD] Found match ignoring spaces: '{existing_file}'")
                found = True
                break
                
            # If still not found, try case insensitive
            if not found and existing_file.lower() == decoded_filename.lower():
                decoded_filename = existing_file
                current_app.logger.info(f"[DOWNLOAD] Found case-insensitive match: '{existing_file}'")
                found = True
                break
    
    # Use the corrected filename path
    file_path = os.path.join(upload_dir, decoded_filename)
    if not os.path.exists(file_path):
        current_app.logger.error(f"[DOWNLOAD] File not found after all matching attempts: '{decoded_filename}'")
        return jsonify({"error": "File not found"}), 404
    
    # Set as_attachment=False to allow browser to preview the file instead of forcing download
    return send_from_directory(
        upload_dir,
        decoded_filename,
        as_attachment=False  # Allow browser to preview the file
    )

@main.route("/graph/render-clusters-merged", methods=["POST"])
def render_semantic_cluster_merge():
    try:
        data = request.get_json()
        current_app.logger.info(f"selected clusters list: {data}", exc_info=True)
        graph = GraphFiles()
        html = graph.render_combined_clusters_to_single_graph(data)

        current_app.graph_builder = graph  

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
        storage = GraphPostgresStorage(dns)
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
        conn = psycopg2.connect(dns)
        cur = conn.cursor()
        cur.execute("""
            SELECT cluster_id, label FROM cluster_labels WHERE graph_id = %s
        """, (graph_id,))
        labels = [{"cluster_id": row[0], "label": row[1]} for row in cur.fetchall()]
        return jsonify(labels)
    except Exception as e:
        current_app.logger.error(f"Error fetching labels: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500