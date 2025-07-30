from crewai import Agent, Task, Crew, Process
from app.llm_model import LLMModel
import os
from pydantic import BaseModel, ValidationError
from typing import List, Optional
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

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
        "Generate a unique, powerful, and highly meaningful label (2-4 words only) for a given cluster of text notes. "
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

logger = logging.getLogger(__name__)

def parse_label_output(raw_output):
    """
    Strictly parse and validate LLM output as LabelOutput using Pydantic. Only accept valid JSON objects.
    Returns label or None if invalid.
    """
    try:
        # If CrewAI returns a CrewOutput object, get its string value
        if hasattr(raw_output, 'content'):
            raw_output = raw_output.content
        # Find the first valid JSON object in the string
        match = re.search(r'\{\s*"label"\s*:\s*".*?"\s*\}', str(raw_output), re.DOTALL)
        if match:
            data = json.loads(match.group(0))
        else:
            # Try to load the whole string as JSON
            data = json.loads(str(raw_output))
        parsed = LabelOutput.parse_obj(data)
        return parsed.label.strip()
    except (json.JSONDecodeError, ValidationError, Exception) as e:
        logger.error(f"Failed to strictly parse LLM output as LabelOutput: {e}\nRaw output: {raw_output}")
        return None

def is_valid_label(label: str) -> bool:
    if not label or not isinstance(label, str):
        return False
    if len(label.strip()) == 0:
        return False
    if len(label.strip().split()) >= 7:  # more than 5 words = too long
        return False
    if len(label.strip()) > 60:  # more than 60 characters = reject
        return False
    # if re.match(r"(?i)^.*label.*|cluster.*|summary.*$", label.strip()):  # label contains placeholder junk
    #     return False
    return True

def generate_unique_label(cluster_text: str, cluster_id: int, existing_labels: set):
    logger.info(f"Starting label generation for Cluster ID: {cluster_id}")
    max_retries = 1
    final_label = None
    for attempt in range(max_retries + 1):
        # Task 1: Label Generation
        generation_task = Task(
            description=(
                "You are to generate a label for the following cluster of text notes. "
                "The label must be exactly 2-4 words, unique among all existing labels, and highly meaningful: it should capture the main semantic theme of the cluster. "
                "Return your answer as a JSON object in the following format: {\"label\": \"...\"}. "
                "Do not include any other text, explanation, or suggestions. Only return the JSON object.\n"
                f"Cluster Content:\n{cluster_text}\n"
                f"Existing Labels: {sorted(existing_labels)}\n"
            ),
            expected_output='{"label": "two-four-word-label"}',
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
            logger.info(f"Raw label output for Cluster ID {cluster_id}: {raw_label}")
            label = parse_label_output(raw_label)
            if not label:
                logger.error(f"No valid label generated for Cluster ID: {cluster_id}. Attempt {attempt+1}/{max_retries+1}")
                cluster_text = cluster_text[len(cluster_text)//2]
                if attempt == max_retries:
                    return f"Cluster {cluster_id}"
                continue
            generated_label = label
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}", exc_info=True)
            generated_label = ""
        if not is_valid_label(generated_label):
            logger.warning(f"⚠️ Invalid label format: {generated_label!r}")
            if attempt == max_retries:
                final_label = f"Cluster {cluster_id}"
                break
            continue
        if generated_label.lower() in existing_labels:
            logger.warning(f"⚠️ Duplicate label: {generated_label!r}")
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

def generate_labels_parallel(clusters: list, existing_labels: set, max_workers: int = 5):
    """
    Generate labels for clusters in parallel, ensuring uniqueness.
    clusters: list of (cluster_text, cluster_id)
    existing_labels: set to track used labels
    max_workers: number of parallel threads
    Returns: dict mapping cluster_id to label
    """
    lock = threading.Lock()
    results = {}
    def label_task(cluster_text, cluster_id):
        # Use a local copy of existing_labels for uniqueness check
        with lock:
            labels_snapshot = set(existing_labels)
        label = generate_unique_label(cluster_text, cluster_id, labels_snapshot)
        # After generation, check for duplicates and update global set
        with lock:
            if label and label.lower() not in existing_labels:
                existing_labels.add(label.lower())
            else:
                # Retry if duplicate, up to 2 more times
                for _ in range(2):
                    label = generate_unique_label(cluster_text, cluster_id, existing_labels)
                    if label and label.lower() not in existing_labels:
                        existing_labels.add(label.lower())
                        break
        return cluster_id, label
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cluster = {
            executor.submit(label_task, cluster_text, cluster_id): cluster_id
            for cluster_text, cluster_id in clusters
        }
        for future in as_completed(future_to_cluster):
            cluster_id, label = future.result()
            results[cluster_id] = label
    return results

def batch_generate_cluster_labels(cluster_texts: list, batch_size: int = 4, max_workers: int = 5):
    """
    Given a list of cluster texts, generate unique labels for each cluster in batches to avoid rate limits.
    Returns a dict mapping cluster_id to label.
    """
    existing_labels = set()
    clusters = []
    for idx, text in enumerate(cluster_texts):
        # Truncate text to half if length > 30,000
        if len(text) > 30000:
            text = text[:len(text)//2]
        clusters.append((text, idx))

    # Log the length of each cluster text
    for text, idx in clusters:
        logger.info(f"Cluster {idx} text length: {len(text)}")
    results = {}
    total = len(clusters)
    for start in range(0, total, batch_size):
        batch = clusters[start:start+batch_size]
        try:
            batch_results = generate_labels_parallel(batch, existing_labels, max_workers=max_workers)
        except Exception as e:
            logger.error(f"Batch label generation failed: {e}")
            batch_results = {idx: generate_unique_label(text, idx, existing_labels) for text, idx in batch}
        results.update(batch_results)
    return results
