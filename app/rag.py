from flask import current_app
from app.chroma_db import ChromaDB
from app.llm_model import LLMModel
from langchain.prompts import PromptTemplate
from app.models.extracted_file_names import ExtractedFilesModel
from langchain.output_parsers import PydanticOutputParser
from flashrank import Ranker, RerankRequest
import re


class RAG:
    def __init__(self, chroma_db: ChromaDB,model_source:str):
        self.crm = chroma_db
        self.model = LLMModel(model_source).model
        current_app.logger.info(f"Model source is {model_source}")
        
        
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
    
    def generate_prompt(self, user_role, curated_query, lida):
      
        prompt = f"""
        You are Solix AI Assistant, an AI assistant designed to help users analyze, interpret, and extract meaningful insights from the given data. 
        Below is a user query along with relevant context retrieved from enterprise data sources via vector search. Use the context to generate a clear and accurate response.

        Instructions:
        Focus on delivering clear, relevant, and detailed information directly related to the query.  
        If some details are missing, provide the most accurate and informative answer possible based on the available data.  
        Offer guidance or suggest next steps if the provided information may not fully address the query.

        Query: "{curated_query}"
        """

        if not lida:
            prompt += """
           - Based on the data, generate visualizations **only if they add value** to the understanding of the data or insights. 
        - If the response involves **numerical data**, **trends**, **distributions**, or other comparable data that would benefit from a **graphical representation** (such as a chart, graph, or plot), generate the python code using libraries like `matplotlib`, `plotly`, or others.
        - **Do not** generate pyhon code if the data is more straightforward where a chart or graph would not provide additional insights. 
        - **Do not** include any extra information like instructions or explanations about how the Python code will generate the chart (e.g., "This code will generate a chart").
        - Make a **data-driven decision** like an analyst: generate visualizations/python code **only** when they are necessary to convey the most important aspects of the data or insights. For example, do not visualize simple or categorical data unless there is a trend or pattern worth highlighting.

            """
        return prompt


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
        return (response.content, curated_query, files,metadata_only)
    
    def process_user_query_screen2(self, query, access_level, user_role, lida):
    
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
        reranked_docs = self.rerank_documents(extracted_data, query)
        top_reranked_docs = reranked_docs[:10]

        prompt = self.generate_prompt(user_role, curated_query,lida = lida)


        context = f"""
            Data: {top_reranked_docs}
            Instruction: {prompt}

            Please use the above Data and Instruction to answer the question.

        """
        

        # Invoke the language model with the constructed prompt
        response = self.model.invoke(context)
        # response_text = response.content

        # #remove python code from the response
        # pattern = r'```python(.*?)```'
        # if re.search(pattern, response_text, re.DOTALL):
        #     # Python code exists, remove it
        #     cleaned_response = re.sub(pattern, '', response_text, flags=re.DOTALL).strip()
        # else:
        #     # No python code, keep original
        #     cleaned_response = response_text.strip()

        # current_app.logger.info(response)       
        return (response.content,curated_query,top_reranked_docs)

    
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
