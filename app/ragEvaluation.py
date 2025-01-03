import opik

#opik.configure(use_local=True)
import opik

#opik.configure(use_local=True)
from opik.evaluation.metrics import Hallucination, AnswerRelevance, Moderation
# from ollama_model_opik import get

class ragEval:
    def __init__(self, ip=None, op=None, context=None):
        self.input = ip
        self.output = op
        self.context = context
        # self.modelO= get(
        # model_name="ollama/llama3.1",
        # api_base="http://192.168.1.116:11434",
        # )


    def update_context(self, ip, op, context):
        self.input = ip
        self.output = op
        self.context = context
        

    def hallucinationFunc(self):
        metric = Hallucination()
        haResult = metric.score(
            input=self.input,
            output=self.output,
            context=self.context
        )
        print("Hallucination: ")
        print(haResult)
        
        print("Hallucination: ")
        print(haResult)
        
        return (haResult.value,haResult.reason)

    def answerRelevanceFunc(self):
        metric = AnswerRelevance()
        arResult = metric.score(
            input=self.input,
            output=self.output,
            context=self.context
        )
        #print("Answer Relevance: ", arResult.value)


        return (arResult.value,arResult.reason)

    def moderationFunc(self,inp=None):
        metric = Moderation()
        testInput=inp if inp is not None else self.output
        moResult = metric.score(
            input=testInput,
        )




        #print("Moderation: ", moResult.value)
        return (moResult.value,moResult.reason)
