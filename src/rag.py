from chroma_db import ChromaDB
from llm_model import LLMModel
from langchain.prompts import PromptTemplate
from models.extracted_file_names import ExtractedFilesModel
from langchain.output_parsers import PydanticOutputParser
import json


class RAG:
    def __init__(self, chroma_db: ChromaDB):
        self.crm = chroma_db
        self.model = LLMModel().model

    def extract_file_data_and_attributes(self, documents):
        file_data = []
        file_data_attributes = []
        sensitivity = []
        for doc in documents:
            file_data.append(doc['data'])
            attributes_dict = json.loads(doc['attributes'])
            file_data_attributes.append(attributes_dict)
            sensitivity.append(doc['sensitivity'])
        return file_data, file_data_attributes,sensitivity
    

    # def process_user_query(self, query, access_level):
    #     data = self.crm.query_db(query_text= query, user_role = access_level, k= 20)
    #     filtered_file_data, filtered_file_attributes, sensitivity = self.extract_file_data_and_attributes(data)
    #     access_instructions = f"""
    #        You are responding to a query based on the user's access level {access_level}. 
      
    #         Here are the rules to follow:
    #         1. Search the provided data for relevant information that matches the query. If no relevant data is found, return: "No data is found."
    #         2. Only return data where the "sensitivity" level is less than or equal to {access_level} (i.e., {sensitivity} <= {access_level}).
    #         3. If the user is not authorized to access any part of the information, return only: "You are not authorized to access this information."
    #         4. Return the relevant data directly without any explanations, extra details, or references to sensitivity levels or access control policies.

    #         Context Data:{filtered_file_data}

    #         Respond to the query with the relevant information. keep the answer concise"
    #         """
    #     curated_query = f"""
    #         Query: {query}
    #         Context: {filtered_file_data}

    #         Follow these instructions strictly:
    #         {access_instructions}
    #     """
    #     response = self.model.invoke(curated_query)
    #     return (response.content, curated_query)

    def process_user_query(self, query, access_level):
        data_chroma = self.crm.query_db(query_text= query, user_role=access_level, k=40)
        print(data_chroma[0])
        # filtered_file_data, sensitivity, filtered_file_name, retention, file_type = self.extract_file_data_and_attributes(data_chroma)

        extracted_data = []  #actual file data
        extracted_attributes = [] #attributes of file
        metadata_only = []   #metadata of file
        
        for doc in data_chroma:
            # Extract the value of the 'data' key and store it separately
            file_data = doc.pop('data', None)
            
            # Append the file_data to extracted_data list (store the removed data content)
            extracted_data.append(file_data)

            # attributes = doc.pop('attributes', None)
            # extracted_attributes.append(attributes)
            
            # Append the remaining metadata (without 'data' key) to metadata_only list
            metadata_only.append(doc)
            
        
        # Now 'metadata_only' contains all documents without the 'data' key
        # and 'extracted_data' contains only the file data (content) from those documents.
        
        
        access_instructions = f"""
        You are interacting with an administrator who has access level {access_level}. Here are the rules for responding:

        1. **Administrator and above (Access Level 5 and above):**
            - Can access all file metadata (file name, file type, sensitivity, retention) but **cannot** access actual file data.
            - Respond fully using only the metadata (e.g., file name, file type, sensitivity, retention) and do **not** include the data
            or attributes. fields from the file.
            - Do not provide explanations regarding access control policies or details beyond the metadata.

        2. **Other Roles (Access Level less than 5):** 
            - This is not applicable for this screen as it is for administrators only.
            
        Always follow these rules strictly:
        1. Only provide the requested metadata (file name, file type, sensitivity, retention).
        2. If metadata is restricted, return only: "You are not authorized to access this information."
        3. Do not include any actual file data or attributes fields from the result.

        Context Data :  {metadata_only}
        Respond concisely to the following query: {query} """

        curated_query = f"""

        Query: {query}

        Follow these instructions strictly:
        {access_instructions}

        """
        response = self.model.invoke(curated_query)
        files = self.show_files(response.content)
        return (response.content, curated_query, files)
    
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
