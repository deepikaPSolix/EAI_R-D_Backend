from datetime import datetime
import json
import os
from pathlib import Path
import time
from celery import group, shared_task, chain
from flask import current_app
import pandas as pd

from app.chroma_db import ChromaDB
from app.cluster_classify import Classification, Clustering, LabelGenerator
from app.dynamic_extractor import DynamicExtractor
from app.file_processor import AudioFileProcessor, GenericFileProcessor, VideoFileProcessor,ExcelFileProcessor
from app.ragEvaluation import ragEval
from app.ragEvaluationScreenTwo import TextAnalysis
import app.utils as utils
from celery.exceptions import MaxRetriesExceededError

#Postgres
from app.postgres_db import DatabaseManager
import os

def process_file_workflow(files: list):
    files_group = []
    for f in files:
        if utils.is_audio_or_video_file(f):
            # Route audio/video files to GPU queue
            files_group.append(parse_file.s(f).set(queue='gpu_queue'))
        else:
            # Route other files to CPU queue (default)
            files_group.append(parse_file.s(f).set(queue='cpu_queue'))
    parse_files = group(files_group)

    process_files_workflow = chain(
        parse_files,
        generate_labels.s(),
        insert_to_db.s(),
    )

    process_files_workflow.freeze

    evaluation_workflow = chain(
        fileSensitivityEvalutionFunction.si(),
        fileAttributesEvaluationFunction.si(),
        fileClusterEvaluationFunction.si(),
    )

    final_workflow = chain(
        process_files_workflow, 
        evaluation_workflow, 
    )


    final_workflow.apply_async()


    current_app.logger.info(f"Initial Workflow started with task ID:")
    return final_workflow

@shared_task(bind=True, max_retries=3, retry_backoff=False)
def generate_labels(self, data: list):
    try:
        current_app.logger.info(f"Initialised generate_labels")
        if not isinstance(data, list):
            data = [data]
        df = pd.DataFrame(data)
        current_app.logger.info(f"Created df from data.")
        df.to_csv('cache/gen_labels.csv')

        if len(df) < 10:
            df["cluster"] = range(len(df))
            label_generator = LabelGenerator()
            labeled_data = label_generator.generate_labels(df[['file_name', 'cluster', 'chunks']])
            labeled_data.to_csv('cache/labels.csv')
            df['cluster_label'] = labeled_data['cluster_label']
            df.to_csv('cache/result.csv')
            return df.to_dict()

        clustering = Clustering()
        classification = Classification(model_path=current_app.config['ML_DIR_PATH'])
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
        parsed_files.to_csv('cache/result.csv')
        return parsed_files.to_dict()
    except Exception as e:
        current_app.logger.error(str(e))
        self.retry(exc=e, countdown=5)

@shared_task()
def insert_to_db(data):
    try:
        df = pd.DataFrame(data)
        df.to_csv('cache/insert_to_db.csv')
        chroma_db = ChromaDB()
        chroma_db.add_documents(df)
    except Exception as e:
        current_app.logger.error(str(e))

@shared_task(bind=True, max_retries=3, retry_backoff=False, countdown=5)
def parse_file(self, file_path: str):
    file_type = Path(file_path).suffix[1:]
    file_name = Path(file_path).name
    file_size = utils.get_human_readable_file_size(file_path)
    try:
        if file_type == "mp3":
            audio_processor = AudioFileProcessor()
            chunks = audio_processor.process_file(file_path)
        elif file_type == "mp4":
            video_processor = VideoFileProcessor()
            chunks = video_processor.process_file(file_path)
        elif file_type in ["xlsx", "xls"]:
            processor = ExcelFileProcessor()
            if file_type == "xlsx":
                generic_processor_xl = GenericFileProcessor()
                generic_processor_xl.xlsx_ocr_replace(input_path=file_path, output_path=file_path)
            chunks = processor.process_file(file_path)
        else:
            generic_processor = GenericFileProcessor()
            try:
                if file_type == 'docx':
                    generic_processor.docx_ocr_replace(input_path=file_path, output_path=file_path)
                elif file_type == 'pptx':
                    generic_processor.pptx_ocr_replace(input_path=file_path, output_path=file_path)
            except Exception as e:
                current_app.logger.error(str(e))
            chunks = generic_processor.process_file(file_path)

        

        attr_ext = DynamicExtractor()
        attr_res = attr_ext.extract_from_file(chunks)
        current_app.logger.info(f"Extracted attributes from file. {attr_res.model_dump()}")
        data = " ".join([chunk.text for chunk in chunks])
        word_count = len(data.split())
        return {
            'file_name' : file_name, 
            'file_size': file_size, 
            'file_type' : file_type, 
            'chunks': [chunk.text for chunk in chunks], 
            'data' : " | ".join([chunk.text for chunk in chunks]), 
            'attributes' : attr_res.model_dump(), 
            "status": "success", 
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "word_count": word_count
        }
    except Exception as e:
        current_app.logger.error(str(e))
        try:
            self.retry(exc=e, countdown=5)
        except MaxRetriesExceededError:
            # Handle final failure after retries
            return {'file_name' : file_name, 'file_type' : file_type, 'chunks': None, 'attributes' : None, "status": "failed"}



@shared_task(bind=True, max_retries=3, retry_backoff=False, countdown=5)
def parse_graph_file(self, file_path: str):
    file_type = Path(file_path).suffix[1:]
    file_name = Path(file_path).name
    file_size = utils.get_human_readable_file_size(file_path)
    if file_type == "mp3":
        audio_processor = AudioFileProcessor()
        chunks = audio_processor.process_file(file_path)
    elif file_type == "mp4":
        video_processor = VideoFileProcessor()
        chunks = video_processor.process_file(file_path)
    else:
        generic_processor = GenericFileProcessor()
        
        try:
            if file_type == 'docx':
                generic_processor.docx_ocr_replace(input_path=file_path, output_path=file_path)
            elif file_type == 'xlsx':
                generic_processor.xlsx_ocr_replace(input_path=file_path, output_path=file_path)
            elif file_type == 'pptx':
                generic_processor.pptx_ocr_replace(input_path=file_path, output_path=file_path)
        except Exception as e:
            current_app.logger.error(str(e))

        chunks = generic_processor.process_file(file_path)
    return chunks

@shared_task()
def cleanup():
    try:
        utils.delete_files_in_directory(current_app.config['UPLOAD_DIR_PATH'])
        return "Files deleted!"
    except Exception as e:
        current_app.logger.error(str(e))


#=============================================
# Evaluation
#=============================================

@shared_task
def fileAttributesEvaluationFunction():

    file_name='./cache/result.csv'
    df=pd.read_csv(file_name)

    columnsToRead=['file_name','attributes', 'data']
    dataD=df[columnsToRead]

    for index, row in dataD.iterrows():
        # print(i['file_name'])
        # print(i['attributes'])

        backup_directory = os.path.join(os.getcwd(), current_app.config['UPLOAD_DIR_PATH'], row['file_name'])
        content=row['data']
        # if os.path.isfile(backup_directory):
        #     with open(backup_directory,'r') as file:
        #         content=file.read()
        #         # print(" the contents of the file")
        #         # print(content)
            
        if not content=='':
            query_input='''
            Extract attributes described in the output format from the text below.
            - If you can't find an attribute, just leave it blank. Do not put null.
            - If there are multiple values for the same attribute, select the most relevant one based on the context provided.
            - Make sure to check is the text is medical data.
            '''
            output=json.loads(row['attributes'].replace("'", '"'))['attributes']
            context=content
            combined_data=[query_input,output,context]
            time.sleep(3)
            evaluationFunction.delay(combinedList=combined_data,include_relevance=False,include_hallucination=True,include_moderation=False,evaluation_result_file="fileAttributesResult.json",evaluation_result_csv="fileAttributesResult.csv",fileName=row['file_name'])


    return "Done with fileAttributesEvaluationFunction!!! "
        

@shared_task
def fileClusterEvaluationFunction():

    file_name='./cache/result.csv'
    df=pd.read_csv(file_name)

    columnsToRead=['file_name','attributes', 'data', 'cluster', 'cluster_label']
    dataD=df[columnsToRead]


    inputQ='''Task: Cluster Document Analysis

            You are provided with a set of documents in a cluster. Perform the following tasks accurately:

            **Cluster Labeling**: Generate a clear and specific label that accurately represents the cluster. Return only the cluster name.

        '''
    
    for cluster in dataD['cluster'].unique():
        cluster_samples = dataD[dataD['cluster'] == int(cluster)]
        attributes = {"attributes":[{
            "label": cs['cluster_label'],
            "file_name": cs['file_name']
        } for i, cs in cluster_samples.iterrows()]}
       
        combinedList=[[inputQ], [attributes], [" ".join(cluster_samples['data'].tolist())]]
        time.sleep(7)
        evaluationFunction.delay(combinedList,include_moderation=False,evaluation_result_file="fileClusterResult.json",evaluation_result_csv="fileClusterResult.csv")

            

    return "Done fileClusterEvaluationFunction!!"

@shared_task
def fileSensitivityEvalutionFunction():
    file_name='./cache/result.csv'
    df=pd.read_csv(file_name)

    columnsToRead=['file_name','attributes', 'data']

    selectedCols=df[columnsToRead]
    #sensitivityCol=df[['sensitivity']]

    for index, row in selectedCols.iterrows():
        # backup_directory = os.path.join(os.getcwd(), './cache/backup_folder', row['file_name'])
        # content=''
        # if os.path.isfile(backup_directory):
        #     with open(backup_directory,'r') as file:
        #         content=file.read()
        attributes = json.loads(row['attributes'].replace("'", '"'))
        content = row['data']
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
        outputRow=attributes['sensitivity']

        context=content
        combinedList=[inputPrompt,outputRow,context]
        evaluationFunction.delay(combinedList=combinedList,include_relevance=True,include_hallucination=False,include_moderation=False,evaluation_result_file="sensitivityEvaluation.json",fileName=row['file_name'])
        time.sleep(5)


    

    return "sensitivity classification Evaluation done"



@shared_task
def evaluationFunction(combinedList,include_relevance=True, include_hallucination=True,include_moderation=True,evaluation_result_file="noEvaluationFileProvided.json",evaluation_result_csv="noEvaluationCSVProvided.csv",fileName=None):
    rag_evaluator=ragEval()

    if len(combinedList)==6:
        curated_query, response, top_reranked_docs,originalQuery, model_name, access_level = combinedList
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

    if len(combinedList) == 6:
        evaluation_data['original_query'] = originalQuery


    # Append the new evaluation data to the existing list
    data.append(evaluation_data)

    # Save the updated list back to the file
    with open("./cache/"+evaluation_result_file, 'w') as file:
        json.dump(data, file, indent=4)   

    db = DatabaseManager()  

    current_app.logger.info("evaluation_result_file: %s", evaluation_result_file)

    # Branch 1: Query Evaluation
    if 'original_query' in evaluation_data:
        role=" "     
        access_level=int(access_level)
        if access_level == 5:
            role = "Administrator"
        elif access_level == 6:
            role = "Data Governance Officers"
        elif access_level == 7:
            role = "Compliance and Legal Teams"
        else:
            role = " "
        MODEL_LABELS = {
            "llama4":      "Meta Llama 4",
            "together":    "Meta Llama 3",
            "openai":      "GPT-4o",
            "qwen":        "Qwen 2.5",
            "ollama":      "Llama3 70B local",
            "qwen_local":  "Qwen 2.5 local",
            "llama3b":     "Llama 3B local",
            "llama8b":     "Llama 8B local"
            }
        model_name_raw = model_name.lower()
        model_label = MODEL_LABELS.get(model_name_raw, model_name)
        query_response_data = {
            "query": evaluation_data.get("original_query"),
            "response": evaluation_data.get("inital_llm_response"),
            "chatbot": "Solix Governance Assistant",
            "model_for_response_generation": model_label,
            "evaluation_method": "LLM_as_Judge-Opik",  
            "hallucination": evaluation_scores.get("hallucination", {}).get("score"),
            "hallucination_reason": evaluation_scores.get("hallucination", {}).get("reason"),
            "relevance": evaluation_scores.get("answer_relevance", {}).get("score"),
            "relevance_reason": evaluation_scores.get("answer_relevance", {}).get("reason"),
            "created_by": role,
            "modified_by": role
        }
        try:
            db.add_query_response(query_response_data)
            current_app.logger.info("Query response evaluation inserted into PostgreSQL.")
        except Exception as e:
            current_app.logger.info("Error inserting query response evaluation into PostgreSQL:", e)

    # Branch 2: Cluster Evaluation – use the evaluation result filename to decide if this is a cluster evaluation
    elif evaluation_result_file == "fileClusterResult.json":
        current_app.logger.info("→ Entering CLUSTER branch")
        # Here, we expect that combinedList[1] is a list containing one dictionary with the key "attributes"
        if isinstance(combinedList[1], list) and len(combinedList[1]) > 0 and isinstance(combinedList[1][0], dict):
            attributes_data = combinedList[1][0].get("attributes", [])
        else:
            attributes_data = []
        if attributes_data and isinstance(attributes_data, list):
            data_category = attributes_data[0].get("label")  # Assume cluster label is consistent for the group
            file_list = [attr.get("file_name") for attr in attributes_data if attr.get("file_name")]
            num_files = len(file_list)
        else:
            data_category = None
            file_list = []
            num_files = 0

        cluster_data = {
            "data_category": data_category,
            "num_files": num_files,
            "file_list": file_list,
            "relevance": evaluation_scores.get("answer_relevance", {}).get("score", 0.0),
            "hallucination": evaluation_scores.get("hallucination", {}).get("score", 0.0)
        }
        try:
            db.add_cluster(cluster_data)
            print("Cluster evaluation inserted/updated into PostgreSQL.")
        except Exception as e:
            print("Error inserting cluster evaluation into PostgreSQL:", e)

    # Branch 3: Standard File Evaluation
    else:
        current_app.logger.info("→ Entering FILE branch")
        file_name_value = evaluation_data.get("file_name")
        if file_name_value:
            file_name_value = file_name_value.strip()
        file_evaluation_data = {
            "file_name": file_name_value,
            "hallucination": evaluation_scores.get("hallucination", {}).get("score"),
            "relevance": evaluation_scores.get("answer_relevance", {}).get("score"),
            "created_by": "system",
            "modified_by": "system"
        }
        try:
            db.add_file_evaluation(file_evaluation_data)
            print("File evaluation inserted/updated into PostgreSQL.")
        except Exception as e:
            print("Error inserting file evaluation into PostgreSQL:", e)




    
    # *******************************************************   
    

    return f"Done with Eval function!!{evaluation_result_file}!!!!"

def make_reasons(relevance_score: float, hallucination_score: float):
    # Relevance Reason
    if relevance_score is None:
        relevance_reason = "Relevance could not be calculated."
    elif relevance_score >= 75:
        relevance_reason = (
            f"The response is highly relevant (score: {relevance_score:.1f}%), "
            "closely matching the user’s query intent."
        )
    elif relevance_score >= 40:
        relevance_reason = (
            f"The response appears to address the query, but the similarity score is moderate "
            f"(score: {relevance_score:.1f}%). This may be due to the embedding‐based measure "
            "underestimating semantic alignment."
        )
    else:
        relevance_reason = (
            f"The similarity score is low (score: {relevance_score:.1f}%), "
            "even though the content looks correct. Low cosine‐similarity can occur when wording "
            "differs substantially from the context embeddings."
        )

    # Hallucination Reason
    if hallucination_score is None:
        hallucination_reason = "Hallucination could not be calculated."
    elif hallucination_score <= 10:
        hallucination_reason = (
            f"Minimal hallucination detected (score: {hallucination_score:.1f}%). "
            "Almost all content is grounded in the context."
        )
    elif hallucination_score <= 30:
        hallucination_reason = (
            f"Moderate hallucination detected (score: {hallucination_score:.1f}%). "
            "Some tokens were not found in the context—this may be due to paraphrasing or synonyms."
        )
    else:
        hallucination_reason = (
              f"High hallucination detected (score: {hallucination_score:.1f}%). "
            "The response introduces creative or novel information beyond the context, which can provide fresh insights but should be verified."
        )

    return relevance_reason, hallucination_reason


@shared_task
def screen2EvaluationFunction(combinedList, evaluation_result_file="queryEvaluationScreen2Result.json"):
    """
    This function evaluates the relevance and hallucination scores for a given response
    and context list and saves the results in a JSON file.
    """
    if len(combinedList)==5:
        response,context_list,originalQuery, model_name, user_role=combinedList

    elif len(combinedList)==3:
        response,context_list,originalQuery=combinedList
    else:
        response,context_list=combinedList

    try:
        # Calculate relevance and hallucination scores
        text_analyzer = TextAnalysis()
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

        if len(combinedList) in (3, 4):
            evaluation_data['original_query'] = originalQuery

       

        # Append the new evaluation data to the existing list
        data.append(evaluation_data)

        # Save the updated list back to the file
        with open("./cache/" + evaluation_result_file, 'w') as file:
            json.dump(data, file, indent=4)

        db = DatabaseManager()

       
        MODEL_LABELS = {
            "llama4":      "Meta Llama 4",
            "together":    "Meta Llama 3",
            "openai":      "GPT-4o",
            "qwen":        "Qwen 2.5",
            "ollama":      "Llama3 70B local",
            "qwen_local":  "Qwen 2.5 local",
            "llama3b":     "Llama 3B local",
            "llama8b":     "Llama 8B local",
            }
        model_name_raw = model_name.lower()
        model_label = MODEL_LABELS.get(model_name_raw, model_name)

        USER_ROLE_LABELS = {
            "patient": "User 1",
            "nurse":   "User 2",
            "doctor":  "User 3",
        }

        mapped_role = USER_ROLE_LABELS.get(user_role.lower(), user_role)

        # First, pull out your numeric scoresF
        rel_score = evaluation_scores.get("relevance_score")
        hall_score = evaluation_scores.get("hallucination_score")

        # Now call your helper:
        relevance_reason, hallucination_reason = make_reasons(rel_score, hall_score)

        query_response_data = {
            "query": originalQuery, 
            "response": response,
            "chatbot": "Solix Governance GPT",
            "model_for_response_generation": model_label,
            "evaluation_method":"Cosine Similarity",
            "hallucination": evaluation_scores.get("hallucination_score"),
            "hallucination_reason": hallucination_reason, 
            "relevance": evaluation_scores.get("relevance_score"),
            "relevance_reason": relevance_reason,
            "created_by": mapped_role,
            "modified_by": mapped_role
        }
        try:
            db.add_query_response(query_response_data)
            print("Screen 2 evaluation record for GPT inserted into PostgreSQL.")
        except Exception as e:
            print("Error inserting Screen 2 evaluation record for GPT:", e)
        # -------- End of New Section --------          

        return f"Done with Screen 2 Evaluation function! Results saved to {evaluation_result_file}."

    except Exception as e:
        print(f"Error in Screen 2 Evaluation Function: {e}")
        return f"Error in Screen 2 Evaluation Function: {str(e)}"
