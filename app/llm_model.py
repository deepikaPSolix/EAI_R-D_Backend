from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_together import ChatTogether
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
import logging
import os

class LLMModel:
    def __init__(self, model_source: str = "together"):
        self.model_source = model_source
        self.model = self._initialize_model()

    @classmethod
    def from_together(cls) -> 'LLMModel':
        return cls("together")

    def _initialize_model(self):
        if self.model_source == "together":
            return ChatTogether(
                temperature=0.1, 
                model='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'
            )
        else:
            base_url = os.getenv("OLLAMA_BASE_URL", "http://10.1.161.62:11435")
            return ChatOllama(
                temperature=0.1, 
                model='llama3.1:70b', 
                base_url=base_url,
                format='json'
            )

    def infer_model(self, query: str, data_model: BaseModel) -> BaseModel:
        """
        Infer the model with the given query and data model.
        
        :param query: The query string to be processed by the model.
        :param data_model: The Pydantic model to parse the output.
        :return: Parsed output as a Pydantic model.
        """
        output_parser = PydanticOutputParser(pydantic_object=data_model)
        format_instructions = output_parser.get_format_instructions()

        prompt = PromptTemplate(
            template="Answer the user query.\n{format_instructions}\n{query}\n",
            input_variables=["query"],
            partial_variables={"format_instructions": format_instructions},
        )
        chain = prompt | self.model | output_parser
        try:
            res = chain.invoke({"query": query})
            return res
        except Exception as e:
            logging.error(f"Error during model inference: {e}")
            raise