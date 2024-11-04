from chroma_db import ChromaDB
from llm_model import LLMModel
from langchain.prompts import PromptTemplate
from models.extracted_file_names import ExtractedFilesModel
from langchain.output_parsers import PydanticOutputParser
import json

from flashrank import Ranker, RerankRequest
from langchain import hub


class RAG:
    def __init__(self, chroma_db: ChromaDB):
        self.crm = chroma_db
        self.model = LLMModel().model
        
    def query_rewriting(self, query):
        imp_instructions = f'''
        Your task is to optimize the given query for semantic search in a vector database.

    - Correct any misspellings or grammatical errors.
    - Remove unnecessary details, making the query as straightforward as possible.
    - Do not output any prefix or suffix; just the rewritten query
        '''
        response = self.model.invoke(f'''  Query: {query}
        Instructions: {imp_instructions} ''')
        curated_query = response.content
        return curated_query
    
    def rerank_documents(self, initial_docs, query):
        ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/opt")
        rerankrequest = RerankRequest(query=query, passages=initial_docs)
        reranked_docs = ranker.rerank(rerankrequest)
        return reranked_docs
    
    def generate_prompt(self, user_role, curated_query):
        access_instructions = ""
        if user_role == "Doctor":
            access_instructions = f"""

            Query: "{curated_query}"

            1.You are assisting as a Doctor with full access to detailed medical information, including diagnosis, treatment history, medical history, lab results, prescriptions, and imaging.
            Respond with comprehensive medical information, including specific test values, diagnoses, and any relevant clinical context to support medical decision-making."

            2.do not provide any sensitive patient information, including SSN, bank details, or other personal identifiers, even if requested."

            3.***In case there are no files or data you can find relevant to the given query, do not output anything and do not hallucinate***.
            """
            return access_instructions


        elif user_role == "Patient":
            access_instructions = f"""

            Query: "{curated_query}"
            1. you are assisting a Patient, you have access to patient's personal health summary, recent diagnoses, treatment plans, prescribed medications, and instructions from healthcare providers.
            2. Avoid technical details or internal doctor discussions.
            3.***In case there are no files or data you can find relevant to the given query, do not output anything and do not hallucinate***.
                    """
            return access_instructions

        elif user_role == "Nurse":
            access_instructions = f"""
                Query: "{curated_query}"
                1.you are assisting a nurse have access to medical records and are focused on monitoring and patient care. Summarize the medical information
                to assist with patient monitoring and follow-up, without clinical decision-making details, and provide any necessary patient care instructions
                or next steps.
                2.Do not provide any sensitive patient information, including SSN, bank details, or other personal identifiers, even if requested.
                3.***In case there are no files or data you can find relevant to the given query, do not output anything and do not hallucinate***.
                """
            return access_instructions

        else:
            return "Invalid user role."

    def process_user_query(self, query, access_level):
        data_chroma = self.crm.query_db(query_text= query, user_role=access_level, k=40)

        extracted_data = []  #actual file data
        metadata_only = []   #metadata of file
        
        for doc in data_chroma:
            # Extract the value of the 'data' key and store it separately
            file_data = doc.pop('data', None)
            doc.pop('attributes', None)
            # Append the file_data to extracted_data list (store the removed data content)
            extracted_data.append(file_data)
            # Append the remaining metadata (without 'data' key) to metadata_only list
            metadata_only.append(doc)
            
        curated_query=self.query_rewriting(query)
        
        
        access_instructions = f"""
        Your task is to respond to the query based solely on the provided  Context Data, adhering strictly to the following guidelines:

        1. **Direct Response Requirement**:
            - For queries requesting specific lists of files **only output the relevant filenames in a simple, numbered list** without 
            additional explanations or details.
            - Do not categorize or provide sector-specific headings or elaborations; simply list the filenames matching the query.


        2. **Response Guidelines**:
            - Avoid any additional text or descriptions beyond the list of filenames.

        3. ***In case there are no files or data you can find relevant to the given query, do not output anything and do not hallucinate***.

        **Context Data**: {metadata_only}

        **Query**: {curated_query}

        The output must only contain the requested response. Do not include any prefix or suffix to the output.

        """
        
        response = self.model.invoke(access_instructions) 
        files = self.show_files(response.content)
        return (response.content, curated_query, files)
    
    def process_user_query_screen2(self, query, access_level, user_role):
    
        curated_query=self.query_rewriting(query)

        data_chroma = self.crm.query_db(query_text=curated_query, user_role=access_level, k=40)
        extracted_data = []
        for doc in data_chroma:

            new_doc = {
                'text': doc['data'],
                'file_name': doc['file_name']
            }

            extracted_data.append(new_doc)
        # reranked docs is a list, which has list of documents.
        reranked_docs = rerank_documents( self, extracted_data, query)
        top_reranked_docs = reranked_docs[:10]

        prompt = generate_prompt(self, user_role, curated_query)


        context = f"""
            Data: {top_reranked_docs}
            Instruction: {prompt}
            Please use the above Data and Instruction to answer the question.
        """


        # Invoke the language model with the constructed prompt
        response = self.model.invoke(context)
        return response.content

    
    def show_files(self, rag_response):
        """
        Uses LLM to extract a list of file names from a textual RAG response.

        Args:
            rag_response (str): The RAG response containing a list of files.

        Returns:
            list: A list of extracted file names.
        """
        prompt_template = PromptTemplate(
            template="""
            - Extract any file names present in the following text and provide only the JSON list below. 
            - If no file names are present, return an empty list. 
            - Do not add any explanation or additional text.


            Here is the text:
            {rag_response}

            Expected output format:
            {{
                "files": ["file1.txt", "file2.txt", ...]
            }}
            """,
            input_variables=["rag_response"]
        )

        output_parser = PydanticOutputParser(pydantic_object=ExtractedFilesModel)
    
        chain = prompt_template | self.model | output_parser
        
        try:
            response = chain.invoke({"rag_response": rag_response})
            return response.files  
        except Exception as e:
            print(f"Error while extracting files: {e}")
            return []
