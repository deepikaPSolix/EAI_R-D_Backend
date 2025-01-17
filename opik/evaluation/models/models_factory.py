from typing import Optional, Any
from . import base_model, litellm_chat_model
from dotenv import load_dotenv
import os

DEFAULT_GPT_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

load_dotenv()
os.environ["TOGETHERAI_API_KEY"] = os.getenv("TOGETHER_API_KEY_OPIK")


def get(model_name: Optional[str], **model_kwargs: Any) -> base_model.OpikBaseModel:
    if model_name is None:
        model_name = DEFAULT_GPT_MODEL_NAME

    return litellm_chat_model.LiteLLMChatModel(model_name=model_name, **model_kwargs)
