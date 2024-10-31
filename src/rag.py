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
