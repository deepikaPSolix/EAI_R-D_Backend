from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import torch
import os


class TextAnalysis:
    def __init__(self, toxicity_model_name="unitary/toxic-bert", embedding_model_name="sentence-transformers/bert-base-nli-mean-tokens"):
        # Load toxicity detection model and tokenizer
        self.toxicity_model = AutoModelForSequenceClassification.from_pretrained(toxicity_model_name)
        self.toxicity_tokenizer = AutoTokenizer.from_pretrained(toxicity_model_name)

        # Load sentence embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # Cache model names for size calculation
        self.toxicity_model_name = toxicity_model_name
        self.embedding_model_name = embedding_model_name

    def calculate_relevance_and_hallucination(self, response, context_list):
        """
        Calculates relevance and hallucination scores between a response and context.
        """
        if not isinstance(context_list, list) or not all(isinstance(entry, str) for entry in context_list):
            raise ValueError("Context must be a list of strings.")

        # Aggregate all string values from the context
        aggregated_context = " ".join(context_list)

        # Generate embeddings for response and aggregated context
        response_embedding = self.embedding_model.encode(response, convert_to_tensor=True).unsqueeze(0)
        context_embedding = self.embedding_model.encode(aggregated_context, convert_to_tensor=True).unsqueeze(0)

        # Calculate semantic similarity (relevance)
        relevance_score = util.cos_sim(response_embedding, context_embedding).item() * 100

        # Token-level hallucination detection using response-context overlap
        response_tokens = set(response.lower().split())
        context_tokens = set(aggregated_context.lower().split())
        hallucinated_tokens = response_tokens - context_tokens

        hallucination_score = (len(hallucinated_tokens) / len(response_tokens)) * 100 if response_tokens else 0

        return relevance_score, hallucination_score, hallucinated_tokens

    def calculate_toxicity(self, text):
        """
        Calculates the toxicity score and token-level reasoning for the given text.
        """
        # Tokenize input
        inputs = self.toxicity_tokenizer(text, return_tensors="pt", truncation=True, return_attention_mask=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Get model outputs
        outputs = self.toxicity_model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        toxic_score = probabilities[0][1].item() * 100  # Toxic class probability

        # Get word importance by calculating gradients
        outputs.logits[0, 1].backward()  # Backpropagate on the toxic class
        gradients = self.toxicity_model.get_input_embeddings().weight.grad[input_ids[0]].abs()
        word_importance = gradients * attention_mask[0].unsqueeze(1)  # Mask out padding tokens
        word_importance = word_importance.sum(dim=1).detach().cpu().numpy()

        # Map word importance to tokens
        tokens = self.toxicity_tokenizer.convert_ids_to_tokens(input_ids[0])
        token_importance = {token: importance for token, importance in zip(tokens, word_importance)}

        # Sort and return reasoning
        reasoning = sorted(token_importance.items(), key=lambda x: x[1], reverse=True)
        return toxic_score, reasoning

