from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_together import ChatTogether
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter


class LLMModel:
    def __init__(self, model_source:str = "ollama"):
        if model_source == "together":
             self.model = ChatTogether(temperature=0.1, model='meta-llama/Llama-3.3-70B-Instruct-Turbo-Free',)
        else:
            self.model = ChatOllama(temperature=0.1, model='llama3.1', base_url="http://192.168.1.116:11434")

    @classmethod
    def from_together(cls):
        return cls("together")

    def _initialize_model(self):
        model = ChatTogether(temperature=0.1, model='meta-llama/Llama-3.3-70B-Instruct-Turbo',)
        # model = ChatOllama(temperature=0.1, model='llama3.1', base_url="http://192.168.1.116:11434")
        self.model = model

    def infer_model(self, query, data_model) -> BaseModel:
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
            print(e)