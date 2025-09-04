from flask import current_app
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_together import ChatTogether
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
import os
import re
import json

class LLMModel:
    def __init__(self, model_source: str = "gpt_oss_120", json_mode: bool = False):
        self.model_source = model_source
        self.json_mode = json_mode
        self.model = self._initialize_model()

    @classmethod
    def from_together(cls) -> 'LLMModel':
        return cls("together")
    
    @classmethod
    def from_openai(cls) -> 'LLMModel':
        return cls("openai")

    @classmethod
    def from_ollama(cls) -> 'LLMModel':
        return cls("ollama")

    def _initialize_model(self):
        if self.model_source == "together":
            return ChatTogether(
                temperature=0.1, 
                model='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'
            )
        elif self.model_source == "openai":
            return ChatOpenAI(
                temperature=0.1, 
                model="gpt-5-nano",
                openai_api_key=os.getenv("OPENAI_API_KEY"),
            )
        elif self.model_source == "qwen":
            return ChatTogether(
                temperature=0.1, 
                model='Qwen/Qwen2.5-Coder-32B-Instruct'
             )
        elif self.model_source == "llama4":
            return ChatTogether(
                temperature=0.1, 
                model='meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8'
             )
        elif self.model_source == "llama3b":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://10.1.161.62:11439")
            return ChatOllama(
                temperature=0.1, 
                model='llama3.2:3b', 
                base_url=base_url,
                format='json' if self.json_mode else ""
            )
        elif self.model_source == "llama8b":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://10.1.161.62:11438")
            return ChatOllama(
                temperature=0.1, 
                model='llama3.1:8b', 
                base_url=base_url,
                format='json' if self.json_mode else ""
            )
        elif self.model_source == "qwen_local":
            base_url = os.getenv("OLLAMA_BASE_URL", "http://10.1.161.62:11437")
            return ChatOllama(
                temperature=0.1, 
                model='qwen2.5-coder:32b', 
                base_url=base_url,
                format='json' if self.json_mode else ""
            )
        elif self.model_source == "gpt_oss_120":
            return ChatTogether(
                temperature=0.1, 
                model='openai/gpt-oss-120b'
            )
        else:
            base_url = os.getenv("OLLAMA_BASE_URL", "http://10.1.161.62:11436")
            return ChatOllama(
                temperature=0.1, 
                model='llama3.1:70b', 
                base_url=base_url,
                format='json' if self.json_mode else ""
            )
    def _clean_json_text(self,text: str) -> str:
        """
        Make model output JSON-loadable:
        - If a ```json fenced block``` exists, use its content.
        - Strip a leading 'json' token before { or [.
        - Extract the first {...} or [...] if extra prose is present.
        """
        if text is None:
            return ""
        t = str(text).strip()

        # Prefer fenced code block content if present
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t, flags=re.IGNORECASE)
        if m:
            t = m.group(1).strip()

        # Drop a leading `json` token before a JSON structure
        t = re.sub(r'^\s*json\s*(?=(\{|\[))', '', t, flags=re.IGNORECASE)

        # Extract first JSON object/array if there is extra text
        m = re.search(r"(\{(?:.|\n)*\}|\[(?:.|\n)*\])", t)
        if m:
            t = m.group(1).strip()

        return t

    def infer_model(self, query: str, data_model: BaseModel) -> BaseModel:
        """
        Run the model and parse JSON safely into the provided Pydantic model.
        """
        output_parser = PydanticOutputParser(pydantic_object=data_model)

        # Stronger instruction: forbid code fences and 'json' prefix
        format_instructions = output_parser.get_format_instructions()
        strict_format_hint = (
            "Return ONLY a valid JSON object. "
            "Do NOT include any markdown fences or the word 'json' before the object."
        )

        prompt = PromptTemplate(
            template="Answer the user query.\n{format_instructions}\n{strict_hint}\n{query}\n",
            input_variables=["query"],
            partial_variables={
                "format_instructions": format_instructions,
                "strict_hint": strict_format_hint
            },
        )

        # Build chain WITHOUT the parser so we can clean/fallback if needed
        chain = prompt | self.model

        try:
            res = chain.invoke({"query": query})
            # Get text regardless of message wrapper
            raw_text = res if isinstance(res, str) else getattr(res, "content", str(res))

            # First attempt: parser on raw text
            try:
                return output_parser.parse(raw_text)
            except Exception:
                # Second attempt: clean and parse
                cleaned = self._clean_json_text(raw_text)
                try:
                    return output_parser.parse(cleaned)
                except Exception:
                    # Last resort: manual JSON -> Pydantic
                    obj = json.loads(cleaned)
                    # pydantic v1 vs v2 compatibility
                    try:
                        return data_model.parse_obj(obj)      # pydantic v1
                    except AttributeError:
                        return data_model.model_validate(obj) # pydantic v2

        except Exception as e:
            current_app.logger.error(f"Error during model inference: {e}", exc_info=True)
            raise

