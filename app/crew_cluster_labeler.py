from crewai import Agent, Task, Crew, Process
from app.llm_model import LLMModel
from flask import current_app

# Load your LLM dynamically from llm_model.py
llm_instance = LLMModel.from_together()  # or from_openai, from_ollama, etc.

label_generator = Agent(
    role="Label Generator",
    goal="Generate a concise and meaningful label for the provided cluster text.",
    backstory="Expert at summarizing textual clusters into short, descriptive labels.",
    llm='gpt-4',
)

label_verifier = Agent(
    role="Label Verifier",
    goal="Check if the proposed label is unique within the provided set of existing labels.",
    backstory="Meticulous verifier, ensuring each label is unique and distinct.",
    llm='gpt-4',
)

label_fixer = Agent(
    role="Label Fixer",
    goal="Generate a revised, unique label when duplicates are detected.",
    backstory="Specialist at refining labels, ensuring uniqueness while retaining meaning.",
    llm='gpt-4',
)

def generate_unique_label(cluster_text: str, cluster_id: int, existing_labels: set):
    current_app.logger.info(f"Starting label generation for Cluster ID: {cluster_id}")

    # Task 1: Label Generation
    generation_task = Task(
        description=f"Generate a concise and meaningful label (2-3 words) for the cluster content:\n{cluster_text}",
        expected_output="Short unique label (2-3 words max).",
        agent=label_generator,
    )

    # Run initial generation
    crew_gen = Crew(
        agents=[label_generator],
        tasks=[generation_task],
        process=Process.sequential,
        verbose=True,
    )
    generated_label = str(crew_gen.kickoff()).strip()
    current_app.logger.info(f"Generated label '{generated_label}' for Cluster ID: {cluster_id}")

    # Task 2: Label Verification
    verifier_task = Task(
        description=f"Check if the label '{generated_label}' is unique among existing labels: {existing_labels}. Respond 'UNIQUE' or 'DUPLICATE'.",
        expected_output="'UNIQUE' or 'DUPLICATE'",
        agent=label_verifier,
    )

    crew_verify = Crew(
        agents=[label_verifier],
        tasks=[verifier_task],
        process=Process.sequential,
        verbose=True,
    )
    verification_result = str(crew_verify.kickoff()).strip()
    current_app.logger.info(f"Verification result for label '{generated_label}': {verification_result}")

    # If duplicate, fix the label
    if verification_result.upper() == 'DUPLICATE':
        fixer_task = Task(
            description=(
                f"The previously generated label '{generated_label}' was a duplicate."
                f" Generate a new improved and unique label (2-3 words max) for the cluster:\n{cluster_text}"
            ),
            expected_output="Short improved and unique label (2-3 words max).",
            agent=label_fixer,
        )

        crew_fix = Crew(
            agents=[label_fixer],
            tasks=[fixer_task],
            process=Process.sequential,
            verbose=True,
        )
        final_label = str(crew_fix.kickoff()).strip()
        current_app.logger.info(f"Improved unique label generated: '{final_label}' for Cluster ID: {cluster_id}")
    else:
        final_label = generated_label

    existing_labels.add(final_label.lower())  # Track the new label as used
    return final_label
