from langchain_together import ChatTogether
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate


class LLMModel:
    def __init__(self):
        self._initialize_model()

    def _initialize_model(self):
        model = ChatTogether(temperature=0.1)
        self.model = model

    def infer_model(self, query, data_model):
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
            return None