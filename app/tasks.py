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
from app.file_processor import AudioFileProcessor, GenericFileProcessor, VideoFileProcessor
from app.ragEvaluation import ragEval
from app.ragEvaluationScreenTwo import TextAnalysis
from app.utils import delete_files_in_directory, is_audio_or_video_file
from celery.exceptions import MaxRetriesExceededError

def process_file_workflow(files: list):
    files_group = []
    for f in files:
        if is_audio_or_video_file(f):
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

    final_workflow = chain(process_files_workflow, evaluation_workflow, cleanup.si())


    final_workflow.apply_async()


    current_app.logger.info(f"Initial Workflow started with task ID:")
    return final_workflow

@shared_task(bind=True, max_retries=3, retry_backoff=False)
def generate_labels(self, data: list):
    try:
        df = pd.DataFrame(data)
        df.to_csv('cache/gen_labels.csv')
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
        return {'file_name' : file_name, 'file_type' : file_type, 'chunks': [chunk.text for chunk in chunks], 'data' : " | ".join([chunk.text for chunk in chunks]), 'attributes' : attr_res.model_dump(), "status": "success"}
    except Exception as e:
        current_app.logger.error(str(e))
        try:
            self.retry(exc=e, countdown=5)
        except MaxRetriesExceededError:
            # Handle final failure after retries
            return {'file_name' : file_name, 'file_type' : file_type, 'chunks': None, 'attributes' : None, "status": "failed"}

@shared_task()
def cleanup():
    try:
        delete_files_in_directory(current_app.config['UPLOAD_DIR_PATH'])
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

@shared_task
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

