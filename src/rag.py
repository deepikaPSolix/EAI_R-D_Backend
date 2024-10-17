from chroma_db import ChromaDB
from llm_model import LLMModel
import json


class RAG:
    def __init__(self, chroma_db: ChromaDB):
        self.crm = chroma_db
        self.model = LLMModel().model

    def extract_file_data_and_attributes(self, documents):
        file_data = []
        file_data_attributes = []
        for doc in documents:
            file_data.append(doc['data'])
            attributes_dict = json.loads(doc['attributes'])
            file_data_attributes.append(attributes_dict)
        return file_data, file_data_attributes
    

    def process_user_query(self, query, access_level):
        data = self.crm.query_db(query_text= query, user_role = access_level, k= 10)
        filtered_file_data, filtered_file_attributes = self.extract_file_data_and_attributes(data)
        access_instructions = f"""
            You are processing a query from a user with access level {access_level}.
            Each document in the context has a sensitivity level, and you are only allowed to retrieve and display data from documents with a sensitivity level
            that is less than or equal to the user's access level which is {access_level}.
            Instructions:
            1. Search the provided 'file_data' and 'file_attributes' for any relevant information that matches the query.
            2. Focus on all aspects of the query, including Patient ID, Diagnosis, Exam results, Provider details, and other medical attributes.
            3. If the query asks for any information that falls outside the user's access level, respond only with: "You are not authorized to access this information."
            4. Return the exact data requested (e.g., Diagnosis, Exam findings) if the user is authorized. Do not provide explanations, extra details, or partial information.
            5. If the user has access to the requested information, respond with just the relevant details from both 'file_data' and 'file_attributes'.
            Context Data:
            Attributes: {filtered_file_attributes}
            Data: {filtered_file_data}
            Respond to the query based on the user's access level and the context above.
            If the query involves data beyond the user's access level, return only: "You are not authorized to access this information."
            Please dont give me extra information. just the response or return only "You are not authorized to access this information."
        """
        response = self.model.invoke(f"""
            Query: {query}
            Context: {filtered_file_data}

            Follow these instructions strictly:
            {access_instructions}
        """)
        return response.content