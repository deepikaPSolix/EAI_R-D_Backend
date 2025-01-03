from dotenv import load_dotenv
from flask import Flask, request, jsonify
import os
from chroma_db import ChromaDB
from cluster_classify import ClusterAndClassify
from attribute_extractor import AttributeExtractor
from doc_file import DocFile
import pandas as pd
from celery import Celery, chain,shared_task
from chromadb.errors import NotFoundError
from rag import RAG
from app.ragEvaluation import ragEval
import numpy as np
import time
import json
import shutil
from app.ragEvaluationScreenTwo import TextAnalysis

app = Flask(__name__)

rag_evaluator=ragEval()

text_analyzer = TextAnalysis()

# Configure Celery
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'], backend=app.config['CELERY_RESULT_BACKEND'])


# Ensure the folder for saving uploaded files exists
UPLOAD_FOLDER = 'cache/uploads'
ML_FOLDER = 'cache/ml-model'

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



@shared_task
def fileAttributesEvaluationFunction(dataD):
    # print(dataD)

    for i in dataD:
        # print(i['file_name'])
        # print(i['attributes'])

        backup_directory = os.path.join(os.getcwd(), './cache/backup_folder', i['file_name'])
        content=''
        if os.path.isfile(backup_directory):
            with open(backup_directory,'r') as file:
                content=file.read()
                # print(" the contents of the file")
                # print(content)
            
        if not content=='':
            query_input='''
            Extract attributes described in the output format from the text below.
            - If you can't find an attribute, just leave it blank. Do not put null.
            - If there are multiple values for the same attribute, select the most relevant one based on the context provided.
            - Make sure to check is the text is medical data.
            '''
            output=i['attributes']
            context=content
            combined_data=[query_input,output,context]
            time.sleep(3)
            evaluationFunction.delay(combinedList=combined_data,include_relevance=False,include_hallucination=True,include_moderation=False,evaluation_result_file="fileAttributesResult.json",evaluation_result_csv="fileAttributesResult.csv",fileName=i['file_name'])


    return "Done with fileAttributesEvaluationFunction!!! "
        

@shared_task
def fileClusterEvaluationFunction():

    folder_path = "./cache/cluster_files"
    visited_files = set() 
    questions_list = []
    results_dict = {}

    inputQ='''Task: Cluster Document Analysis

            You are provided with a set of documents in a cluster. Perform the following tasks accurately:

            1. **Cluster Labeling**: Generate a clear and specific label that accurately represents the cluster. Return only the cluster name.

            2. **Sensitivity Classification**: For each document, assign a sensitivity level (integer) based on the following categories:
            - 1: Public Data (e.g., public reports, statistics)
            - 2: Internal Data (internal use, not for external sharing)
            - 3: Confidential Data (personal or sensitive information)
            - 4: Restricted Data (highly sensitive, access limited)
            - 5: Private Data (personal data protected by privacy laws)
            - 6: Critical Data (vital for urgent care or life-saving actions)
            - 7: Regulatory Data (compliance with legal/regulatory rules)

            Ensure documents containing PII, trade secrets, legal, medical, or intellectual property are classified as level 3 or higher. Assign levels 6 or 7 for critical or regulatory data.

            3. **Top Data Types**: Analyze the document and identify 3 to 5 **distinct** and meaningful data types that justify the sensitivity classification. Do not rely on the example data types provided below—these are only for reference. The identified data types must be directly related to the actual content of the document. Return these data types in CSV format. The data types should be relevant to the document's content, and similar types must not be repeated.

            Reference examples (for understanding only, do not use as output unless relevant):
                - PII
                - PHI
                - EHR Data
                - Medical History
                - Lab Results
                - Prescription Data
                - Patient Satisfaction Surveys
                - Appointment Records
                - Demographic Information
                - Contact Information
                - Health Insurance Details
                - Caregiver Information
                - Clinical Trial Data
                - Adverse Event Reports
                - Imaging Data
                - Genetic Information
                - Diagnosis Codes (ICD-10)
                - Treatment Protocols
                - Medical Devices Information
                - Immunization Records
                - Clinical Notes
                - Anonymized Clinical Notes
                - Research Findings
                - Billing Information
                - Insurance Claims Data
                - Compliance Reports
                - Audit Trails
                - Internal Policies
                - Staff Scheduling Information
                - Facility Management Data
                - Equipment Inventory
                - Financial Reports
                - Strategic Plans
                - Billing Disputes
                - Operational Efficiency Metrics
                - Staffing Levels
                - Risk Management Reports


            4. **Data Points**: Provide up to 10 key that contribute to the sensitivity level. Use commas to seperate values.

            5. **Retention Period**: Assign a retention period for each document based on its type, specifying the time in years and months.
        '''
    
    questions_list = []
    results_dict = {}
    
    # Step 1: Group files by their identifiers
    file_map = {}
    for file_name in os.listdir(folder_path):
        # Extract the identifier from the file name
        if file_name.startswith('cluster') and file_name.endswith('.txt'):
            identifier = file_name[len('cluster'):file_name.find('.txt')]
        elif file_name.startswith('qres_cluster') and file_name.endswith('.json'):
            identifier = file_name[len('qres_cluster'):file_name.find('.json')]
        else:
            continue  # Skip files that don't match the patterns

        if identifier not in file_map:
            file_map[identifier] = {}
        if file_name.startswith('cluster'):
            file_map[identifier]['txt'] = file_name
        elif file_name.startswith('qres_cluster'):
            file_map[identifier]['json'] = file_name


    # Step 2: Process files in pairs
    for identifier, files in file_map.items():
        txt_file = files.get('txt')
        json_file = files.get('json')

        if txt_file and json_file:
            # Both files are present; process them together
            with open(os.path.join(folder_path, txt_file), 'r', encoding='utf-8') as file:
                txt_content = file.read().strip()

            with open(os.path.join(folder_path, json_file), 'r', encoding='utf-8') as file:
                json_content = json.load(file)
            
            combinedList=[[inputQ],[json_content],[txt_content]]
            time.sleep(7)
            evaluationFunction.delay(combinedList,include_moderation=False,evaluation_result_file="fileClusterResult.json",evaluation_result_csv="fileClusterResult.csv")

    return "Done fileClusterEvaluationFunction!!"

@shared_task
def fileSensitivityEvalutionFunction():
    file_name='./cache/result.csv'
    df=pd.read_csv(file_name)

    columnsToRead=['file_name','sensitivity']

    selectedCols=df[columnsToRead]
    #sensitivityCol=df[['sensitivity']]

    for index, row in selectedCols.iterrows():
        backup_directory = os.path.join(os.getcwd(), './cache/backup_folder', row['file_name'])
        content=''
        if os.path.isfile(backup_directory):
            with open(backup_directory,'r') as file:
                content=file.read()

        inputPrompt='''What is the Sensivitiy level for this document or cluster:
            Here are the details just for your information
            1. **Sensitivity Classification**: For each document, assign a sensitivity level (integer) based on the following categories:
            - 1: Public Data (e.g., public reports, statistics)
            - 2: Internal Data (internal use, not for external sharing)
            - 3: Confidential Data (personal or sensitive information)
            - 4: Restricted Data (highly sensitive, access limited)
            - 5: Private Data (personal data protected by privacy laws)
            - 6: Critical Data (vital for urgent care or life-saving actions)
            - 7: Regulatory Data (compliance with legal/regulatory rules)

            Ensure documents containing PII, trade secrets, legal, medical, or intellectual property are classified as level 3 or higher. Assign levels 6 or 7 for critical or regulatory data.

            '''
        outputRow=row['sensitivity']

        context=content
        combinedList=[inputPrompt,outputRow,context]
        evaluationFunction.delay(combinedList=combinedList,include_relevance=True,include_hallucination=False,include_moderation=False,evaluation_result_file="sensitivityEvaluation.json",fileName=row['file_name'])
        time.sleep(5)


    

    return "sensitivity classification Evaluation done"

# Heavy processing task
@celery.task
def process_files(files):
    parsed_files = [DocFile(file_path= path) for path in files] # Extract data from file and put in DocFile Object

    # Cluster Data
    cc = ClusterAndClassify()
    cc_res = cc.cluster_classify(data = parsed_files)
    labels = cc.generate_cluster_labels(cc_res)
    labels.to_csv('cache/labels.csv')
    # Extract attributes
    attr_ext = AttributeExtractor()
    attr_res = attr_ext.extract_from_files(parsed_files)
    attr_res.fillna('')

    result = pd.merge(labels, attr_res, on='file_name', how='left') 
    result['attributes'] = result[attr_res.columns.difference(['file_name'])].apply(lambda row: row.to_dict(), axis=1)
    result = result[['file_name', 'data', 'label','sensitivity', 'data_classifiers', 'responsible_values', 'retention_time', 'attributes']]
    result.to_csv('cache/result.csv')
    cdb = ChromaDB()
    res = cdb.add_documents(result)
    if res:
        BACKUP_FOLDER = './cache/backup_folder'
        os.makedirs(BACKUP_FOLDER, exist_ok=True)  # Ensure backup folder exists

        for file_name in os.listdir(UPLOAD_FOLDER):
            source_path = os.path.join(UPLOAD_FOLDER, file_name)
            destination_path = os.path.join(BACKUP_FOLDER, file_name)

            # Check if file exists in BACKUP_FOLDER and if it's a file
            if os.path.isfile(source_path) and not os.path.exists(destination_path):
                shutil.copy2(source_path, destination_path)  # Copy with metadata
                print(f"Copied: {file_name}")
            elif os.path.exists(destination_path):
                print(f"Skipped (already exists): {file_name}")

        delete_files_in_directory(UPLOAD_FOLDER)
    delete_files_in_directory(UPLOAD_FOLDER)


    result_subset=result[['file_name','attributes']]
    result_subset_json = result_subset.to_dict(orient="records")


    workflow = chain(
        fileSensitivityEvalutionFunction.si(),
        fileAttributesEvaluationFunction.si(dataD=result_subset_json),
        fileClusterEvaluationFunction.si()
    )
    
    workflow.apply_async()

    return "Added documents to chromadb!"


@celery.task
def evaluationFunction(combinedList,include_relevance=True, include_hallucination=True,include_moderation=True,evaluation_result_file="noEvaluationFileProvided.json",evaluation_result_csv="noEvaluationCSVProvided.csv",fileName=None):
    

    if len(combinedList)==4:
        curated_query, response, top_reranked_docs,originalQuery = combinedList
    else:
        curated_query, response, top_reranked_docs = combinedList


    
    # print("Rag Eval Result in celery task are :  ")
    # print(response)
    # print(top_reranked_docs)
    rag_evaluator.update_context(curated_query, response, top_reranked_docs)


    def safe_execute(func, *args, **kwargs):
        try:
            result = func(*args, **kwargs)
            # If result is already a tuple of (score, reason), return it directly
            if isinstance(result, tuple) and len(result) == 2:
                return result
            # If result is a single value, return it with None as reason
            return result, None
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            return None, f"Error: {str(e)}"
    
    def retry_with_sleep(func, retries=3, sleep_duration=5, *args, **kwargs):
        last_exception = None
        for attempt in range(retries):
            try:
                result = func(*args, **kwargs)
                # Handle tuple results consistently
                if isinstance(result, tuple) and len(result) == 2:
                    return result
                return result, None
            except Exception as e:
                last_exception = e
                print(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                if attempt < retries - 1:
                    time.sleep(sleep_duration)
        return None, f"Error after {retries} retries: {str(last_exception)}"
    
    def safe_float_convert(value):
        if value is None:
            return None
        try:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, tuple):
                # If it's a tuple, try to convert the first element
                return float(value[0]) if value[0] is not None else None
            return float(value)
        except (TypeError, ValueError):
            return None
    
    if include_relevance:
        relevance_result = safe_execute(rag_evaluator.answerRelevanceFunc)
    
    if include_hallucination:
        hallucination_result = retry_with_sleep(rag_evaluator.hallucinationFunc)
    
    if include_moderation:
        moderation_result = retry_with_sleep(rag_evaluator.moderationFunc, inp=response)

    evaluation_scores={}
    if include_relevance:
        evaluation_scores["answer_relevance"] = {
            "score": safe_float_convert(relevance_result[0]),
            "reason": relevance_result[1]
        }
        
    if include_hallucination:
        evaluation_scores["hallucination"] = {
            "score": safe_float_convert(hallucination_result[0]),
            "reason": hallucination_result[1]
        }
        
    if include_moderation:
        evaluation_scores["moderation"] = {
            "score": safe_float_convert(moderation_result[0]),
            "reason": moderation_result[1]
        }
    

    if not os.path.exists("./cache/"+evaluation_result_file):
        with open("./cache/"+evaluation_result_file, 'w') as file:
            # Initialize the file with an empty JSON array
            json.dump([], file, indent=4)
            print(f"Created new {evaluation_result_file} and initialized with an empty list.")

    # Load existing data from the file
    with open("./cache/"+evaluation_result_file, 'r') as file:
        try:
            data = json.load(file)  # Parse the existing JSON data
        except json.JSONDecodeError:
            data = []  # Fallback to an empty list if JSON is invalid

    # Prepare new evaluation data
    evaluation_data = {
        'query': curated_query,
        'inital_llm_response': response,
        'evaluation_scores': evaluation_scores
    }

    if fileName is not None:
        evaluation_data['file_name'] = fileName

    if len(combinedList) == 4:
        evaluation_data['original_query'] = originalQuery


    # Append the new evaluation data to the existing list
    data.append(evaluation_data)

    # Save the updated list back to the file
    with open("./cache/"+evaluation_result_file, 'w') as file:
        json.dump(data, file, indent=4)

    return f"Done with Eval function!!{evaluation_result_file}!!!!"

@celery.task
def screen2EvaluationFunction(combinedList, evaluation_result_file="queryEvaluationScreen2Result.json"):
    """
    This function evaluates the relevance and hallucination scores for a given response
    and context list and saves the results in a JSON file.
    """


    if len(combinedList)==3:
        response,context_list,originalQuery=combinedList
    else:
        response,context_list=combinedList

    try:
        # Calculate relevance and hallucination scores
        relevance_score, hallucination_score, hallucinated_tokens = text_analyzer.calculate_relevance_and_hallucination(
            response, context_list
        )

        # Prepare evaluation scores
        evaluation_scores = {
            "relevance_score": relevance_score,
            "hallucination_score": hallucination_score
        }

        

        # Ensure the result file exists
        if not os.path.exists("./cache/" + evaluation_result_file):
            with open("./cache/" + evaluation_result_file, 'w') as file:
                # Initialize the file with an empty JSON array
                json.dump([], file, indent=4)
                print(f"Created new {evaluation_result_file} and initialized with an empty list.")

        # Load existing data from the file
        with open("./cache/" + evaluation_result_file, 'r') as file:
            try:
                data = json.load(file)  # Parse the existing JSON data
            except json.JSONDecodeError:
                data = []  # Fallback to an empty list if JSON is invalid

        # Prepare new evaluation data
        evaluation_data = {
            'response': response,
            'context': context_list,
            'evaluation_scores': evaluation_scores
        }

        if len(combinedList) == 3:
            evaluation_data['original_query'] = originalQuery

       

        # Append the new evaluation data to the existing list
        data.append(evaluation_data)

        # Save the updated list back to the file
        with open("./cache/" + evaluation_result_file, 'w') as file:
            json.dump(data, file, indent=4)

        return f"Done with Screen 2 Evaluation function! Results saved to {evaluation_result_file}."

    except Exception as e:
        print(f"Error in Screen 2 Evaluation Function: {e}")
        return f"Error in Screen 2 Evaluation Function: {str(e)}"




@app.route("/")
def home():
    return "<p>Welcome to EAI!!!</p>"


@app.route("/docs/uploadandtrain", methods=['POST'])
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
       
        combinedList=[res[1],res[0],res[3],data["query"]]
        msg=evaluationFunction.delay(combinedList,include_relevance=True,include_hallucination=True,include_moderation=False,evaluation_result_file="queryEvaluationScreen1Results.json",evaluation_result_csv="queryEvaluationScreen1Results.csv")
        

        return jsonify({"response": res[0], "curated_query": res[1], "files": res[2]})
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500
    

@app.route('/rag2/query', methods=["POST"])
def query_rag2():
    try:
        data = request.get_json()
        if data is None:
            raise ValueError("Missing data in the request body")

        rag = RAG(ChromaDB())
        res,curated_query,top_reranked_docs= rag.process_user_query_screen2(data['query'], data["access_level"], data["user_role"])

        formatted_data = [
            f"filename: {item['file_name']}, text: {item['text']}" for item in top_reranked_docs
        ]

        combinedList = [
            str(curated_query), 
            str(res),            
            formatted_data,
            data["query"]    
        ]
        
        ## This is for OPIK
        # message = evaluationFunction.delay(combinedList,evaluation_result_file="queryEvaluationScreen2Result.json",evaluation_result_csv="queryEvaluationScreen2Result.csv")
        # print(message.id)

        ## This is for TextAnalysis class (cosine distance)
        combinedList = [ 
            str(res),            
            formatted_data,
            data["query"]    
        ]
        message=screen2EvaluationFunction(combinedList)
    


        return jsonify({"response": res})
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500


@app.route('/sensitivityEvalApi',methods=["GET"])
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
    
@app.route('/attributesEvalApi',methods=["GET"])
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
    
@app.route('/clusterEvalApi',methods=["GET"])
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
    
@app.route('/rag1EvalApi',methods=["GET"])
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
    

@app.route('/rag2EvalApi',methods=["GET"])
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
        res = db.update_documents(data)
        return jsonify({'status': res})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500
    
@app.route('/docs', methods=['DELETE'])
def delete_files():
    try:
        ChromaDB().delete_all_docs()
        return jsonify({'status': "Deleted all records."})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    