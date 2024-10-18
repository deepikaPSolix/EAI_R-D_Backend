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
        sensitivity = []
        for doc in documents:
            file_data.append(doc['data'])
            attributes_dict = json.loads(doc['attributes'])
            file_data_attributes.append(attributes_dict)
            sensitivity.append(doc['sensitivity'])
        return file_data, file_data_attributes,sensitivity
    

    def process_user_query(self, query, access_level):
        data = self.crm.query_db(query_text= query, user_role = access_level, k= 20)
        filtered_file_data, filtered_file_attributes, sensitivity = self.extract_file_data_and_attributes(data)
        access_instructions = f"""
           You are responding to a query based on the user's access level {access_level}. 
      
            Here are the rules to follow:
            1. Search the provided data for relevant information that matches the query. If no relevant data is found, return: "No data is found."
            2. Only return data where the "sensitivity" level is less than or equal to {access_level} (i.e., {sensitivity} <= {access_level}).
            3. If the user is not authorized to access any part of the information, return only: "You are not authorized to access this information."
            4. Return the relevant data directly without any explanations, extra details, or references to sensitivity levels or access control policies.

            Context Data:{filtered_file_data}

            Respond to the query with the relevant information. keep the answer concise"
            """
        curated_query = f"""
            Query: {query}
            Context: {filtered_file_data}

            Follow these instructions strictly:
            {access_instructions}
        """
        response = self.model.invoke(curated_query)
        return (response.content, curated_query)