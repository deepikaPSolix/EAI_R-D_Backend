from crewai import Agent, Task, Crew, Process,LLM
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

llm = LLM(model="together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
          api_key=os.environ.get("TOGETHER_API_KEY"),
          base_url="https://api.together.xyz/v1"
        )
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
    llm=llm
)

validator_agent = Agent(
    role="Label Validator",
    goal=(
        "Review a proposed cluster label and confirm it is concise (2-4 words), unique in meaning (not just string), and highly relevant to the cluster content. "
        "Reject labels that are too long, generic, duplicate any existing label by meaning, or do not cover the cluster's main theme."
    ),
    backstory=(
        "You are an expert in semantic validation. You ensure all cluster labels are precise, unique in meaning, and meaningful. "
        "You use advanced reasoning to detect semantic duplicates and ensure coverage."
    ),
    llm=llm
)

refiner_agent = Agent(
    role="Label Refiner",
    goal=(
        "Improve a failed or rejected cluster label, making it concise, unique in meaning, and highly relevant to the cluster content. "
        "Never repeat any label from the provided list, and ensure the label covers the whole cluster's main theme."
    ),
    backstory=(
        "You are an expert in semantic refinement. You fix and improve cluster labels that do not meet requirements, using advanced reasoning."
    ),
    llm=llm
)

class LabelOutput(BaseModel):
    label: str

logger = logging.getLogger(__name__)

def parse_label_output(raw_output):
    try:
        if hasattr(raw_output, 'content'):
            raw_output = raw_output.content
        match = re.search(r'\{\s*"label"\s*:\s*".*?"\s*\}', str(raw_output), re.DOTALL)
        if match:
            data = json.loads(match.group(0))
        else:
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
    if len(label.strip().split()) >= 7:
        return False
    if len(label.strip()) > 60:
        return False
    return True

def validate_label(label, cluster_text, existing_labels):
    """
    Use Validator agent to check label validity and semantic uniqueness/coverage.
    Returns (valid, reason).
    """
    validation_task = Task(
        description=(
            f"Review the following label for a cluster of text notes. "
            f"Label: {label}\n"
            f"Cluster Content:\n{cluster_text}\n"
            f"Existing Labels: {sorted(existing_labels)}\n"
            "Respond with a JSON object: {\"valid\": true/false, \"reason\": \"...\"} explaining why the label is valid or not. "
            "Check for semantic uniqueness (not just string match) and coverage of the cluster's meaning."
        ),
        expected_output='{"valid": true, "reason": "..."}',
        agent=validator_agent,
    )
    try:
        crew_val = Crew(
            agents=[validator_agent],
            tasks=[validation_task],
            process=Process.sequential,
            verbose=True,
        )
        raw_val = crew_val.kickoff()
        match = re.search(r'\{"valid"\s*:\s*(true|false),\s*"reason"\s*:\s*".*?"\}', str(raw_val), re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return data["valid"], data["reason"]
        match = re.search(r'\{"valid"\s*:\s*(true|false)\}', str(raw_val))
        if match:
            valid = json.loads(match.group(0))["valid"]
            return valid, None
    except Exception as e:
        logger.error(f"Validator agent failed: {e}")
    return False, None

def refine_label(label, cluster_text, existing_labels):
    logger.info(f"Invoking Refiner agent for label: {label!r}")
    refine_task = Task(
        description=(
            f"Improve the following label for a cluster of text notes. "
            f"Label: {label}\n"
            f"Cluster Content:\n{cluster_text}\n"
            f"Existing Labels: {sorted(existing_labels)}\n"
            "Return your answer as a JSON object: {\"label\": \"...\"}. "
            "Ensure the label is semantically unique and covers the cluster's main theme."
        ),
        expected_output='{"label": "two-four-word-label"}',
        agent=refiner_agent,
    )
    try:
        crew_ref = Crew(
            agents=[refiner_agent],
            tasks=[refine_task],
            process=Process.sequential,
            verbose=True,
        )
        raw_ref = crew_ref.kickoff()
        return parse_label_output(raw_ref)
    except Exception as e:
        logger.error(f"Refiner agent failed: {e}")
    return None

def generate_unique_label(cluster_text: str, cluster_id: int, existing_labels: set):
    logger.info(f"Starting label generation for Cluster ID: {cluster_id}")
    max_retries = 2
    final_label = None
    failed_labels = []
    for attempt in range(max_retries + 1):
        # Generator agent
        generation_task = Task(
            description=(
                "You are to generate a label for the following cluster of text notes. "
                "The label must be exactly 2-4 words, unique among all existing labels (by meaning, not just string), and highly meaningful: it should capture the main semantic theme of the cluster. "
                "Do NOT repeat any label from the provided list of existing labels, even if the meaning is similar. "
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
                failed_labels.append((cluster_text, None))
                continue
            generated_label = label
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}", exc_info=True)
            failed_labels.append((cluster_text, None))
            generated_label = ""
        # Local checks
        if not is_valid_label(generated_label):
            logger.warning(f"⚠️ Invalid label format: {generated_label!r}")
            failed_labels.append((cluster_text, generated_label))
            continue
        # Validator agent
        valid, reason = validate_label(generated_label, cluster_text, existing_labels)
        if not valid:
            logger.warning(f"Validator agent rejected label: {generated_label!r}. Reason: {reason}")
            failed_labels.append((cluster_text, generated_label))
            # Refiner agent
            refined = refine_label(generated_label, cluster_text, existing_labels)
            if refined and is_valid_label(refined):
                valid_refined, reason_refined = validate_label(refined, cluster_text, existing_labels)
                if valid_refined and refined.lower() not in existing_labels:
                    final_label = refined
                    logger.info(f"Refiner agent produced valid label: {refined!r}")
                    break
                else:
                    logger.warning(f"Refined label rejected: {refined!r}. Reason: {reason_refined}")
                    failed_labels.append((cluster_text, refined))
            continue
        if generated_label.lower() in existing_labels:
            logger.warning(f"⚠️ Duplicate label: {generated_label!r}")
            failed_labels.append((cluster_text, generated_label))
            continue
        final_label = generated_label
        break
    # Save label in session memory
    if final_label:
        existing_labels.add(final_label.lower())
    # Audit trail: log all failed attempts
    if failed_labels:
        for fail_text, fail_label in failed_labels:
            logger.info(f"Audit trail: Cluster ID {cluster_id}, Failed label: {fail_label!r}, Text length: {len(fail_text)}")
    return final_label

def generate_labels_parallel(clusters: list, existing_labels: set, max_workers: int = 5):
    lock = threading.Lock()
    results = {}
    def label_task(cluster_text, cluster_id):
        with lock:
            labels_snapshot = set(existing_labels)
        label = generate_unique_label(cluster_text, cluster_id, labels_snapshot)
        with lock:
            if label and label.lower() not in existing_labels:
                existing_labels.add(label.lower())
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
    existing_labels = set()
    clusters = []
    for idx, text in enumerate(cluster_texts):
        if len(text) > 30000:
            text = text[:len(text)//2]
        clusters.append((text, idx))
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