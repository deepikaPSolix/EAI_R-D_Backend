from crewai import Agent, Task, Crew, Process
from app.llm_model import LLMModel
from flask import current_app
import os
from pydantic import BaseModel, ValidationError
from typing import List, Optional
import json
import re
# Load your LLM dynamically from llm_model.py
llm_instance = LLMModel.from_together()  # or from_openai, from_ollama, etc.
dns_host = os.getenv("DNS_HOST")
dns_dbname = os.getenv("DNS_DBNAME")
dns_user = os.getenv("DNS_USER")
dns_password = os.getenv("DNS_PASSWORD")
dns_port = os.getenv("DNS_PORT")
dns = f"host={dns_host} dbname={dns_dbname} user={dns_user} password={dns_password} port={dns_port}"
label_generator = Agent(
    role="Label Generator",
    goal=(
        "Generate a unique, powerful, and highly meaningful label (2-3 words only) for a given cluster of text notes. "
        "The label must capture the core semantic theme of the cluster, be highly relevant to the content, and be distinct from all other labels. "
        "The label should be so clear and descriptive that, if a user later asks a question related to this topic in any way, the system can easily retrieve the right cluster using this label."
    ),
    backstory=(
        "You are an expert in semantic analysis and summarization. "
        "You deeply understand the meaning and context of text clusters, and you excel at distilling their essence into concise, unique, and powerful labels. "
        "You always avoid generic, vague, or repetitive labels. You use advanced semantic reasoning to ensure each label is both unique and maximally informative."
    ),
    llm='gpt-4',
)

class LabelOutput(BaseModel):
    label: str

def parse_label_output(raw_output):
    """Parse and validate LLM output as LabelOutput. Returns label or None if invalid."""
    try:
        if isinstance(raw_output, dict):
            data = raw_output
        else:
            # Find the first JSON object in the string
            match = re.search(r'\{.*\}', str(raw_output), re.DOTALL)
            if match:
                data = json.loads(match.group(0))
            else:
                data = json.loads(raw_output)
        parsed = LabelOutput(**data)
        return parsed.label.strip()
    except (json.JSONDecodeError, ValidationError, Exception) as e:
        current_app.logger.error(f"Failed to parse LLM output as LabelOutput: {e}\nRaw output: {raw_output}")
        return None

def is_valid_label(label: str) -> bool:
    if not label or not isinstance(label, str):
        return False
    if len(label.strip()) == 0:
        return False
    if len(label.strip().split()) > 5:  # more than 5 words = too long
        return False
    if len(label.strip()) > 60:  # more than 60 characters = reject
        return False
    if re.match(r"(?i)^.*label.*|cluster.*|summary.*$", label.strip()):  # label contains placeholder junk
        return False
    return True

def generate_unique_label(cluster_text: str, cluster_id: int, existing_labels: set):
    current_app.logger.info(f"Starting label generation for Cluster ID: {cluster_id}")
    max_retries = 1
    final_label = None
    for attempt in range(max_retries + 1):
        # Task 1: Label Generation
        generation_task = Task(
            description=(
                "You are to generate a label for the following cluster of text notes. "
                "The label must be exactly 2-3 words, unique among all existing labels, and highly meaningful: it should capture the main semantic theme of the cluster. "
                "Return your answer as a JSON object in the following format: {\"label\": \"...\"}. "
                "Do not include any other text, explanation, or suggestions. Only return the JSON object.\n"
                f"Cluster Content:\n{cluster_text}\n"
                f"Existing Labels: {sorted(existing_labels)}\n"
            ),
            expected_output='{"label": "..."}',
            agent=label_generator,
        )
        try:
            crew_gen = Crew(
                agents=[label_generator],
                tasks=[generation_task],
                process=Process.sequential,
                verbose=True,
            )
            raw_label = crew_gen.kickoff()
            current_app.logger.info(f"Raw label output for Cluster ID {cluster_id}: {raw_label}")
            label = parse_label_output(raw_label)
            if not label:
                current_app.logger.error(f"No valid label generated for Cluster ID: {cluster_id}. Attempt {attempt+1}/{max_retries+1}")
                if attempt == max_retries:
                    return f"Cluster {cluster_id}"
                continue
            generated_label = label
        except Exception as e:
            current_app.logger.error(f"❌ Generation failed: {e}", exc_info=True)
            generated_label = ""
        if not is_valid_label(generated_label):
            current_app.logger.warning(f"⚠️ Invalid label format: {generated_label!r}")
            if attempt == max_retries:
                final_label = f"Cluster {cluster_id}"
                break
            continue
        if generated_label.lower() in existing_labels:
            current_app.logger.warning(f"⚠️ Duplicate label: {generated_label!r}")
            if attempt == max_retries:
                final_label = f"Cluster {cluster_id}"
                break
            continue
        final_label = generated_label
        break
    # Save label in session memory
    if final_label:
        existing_labels.add(final_label.lower())
    # Store the label in the database if graph_id is provided
    
    return final_label
