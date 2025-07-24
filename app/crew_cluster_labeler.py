from crewai import Agent, Task, Crew, Process
# import any CrewAI tools if needed (e.g., SerperDevTool for web search, etc.)
labeler_agent = Agent(
    role="Clustering Analyst",
    goal="Generate a concise, descriptive label for a data cluster",
    backstory="You are an AI agent tasked with naming clusters of data points helpfully.",
    llm="gpt-4"  # or your desired model; defaults can be set via env (e.g. OPENAI_MODEL_NAME)
)
cluster_label_task = Task(
    description="Given the data points or description of a cluster: ```{cluster_data}```, generate a short, unique upto 3 word label summarizing the cluster’s theme or content.",
    expected_output="A brief (few words) label that uniquely identifies this cluster.",
    agent=labeler_agent,
    markdown=False  # not needed to format output in Markdown for a simple label
)

cluster_label_crew = Crew(
    agents=[labeler_agent],
    tasks=[cluster_label_task],
    process=Process.sequential,  # tasks run in order (only one task here)
    verbose=True,               # enable verbose logging for debugging (optional)
    memory=False                # we likely don't need persistent memory between runs for labeling
)
def generate_cluster_label(cluster_data: str) -> str:
    # Prepare inputs for the crew (matching placeholders in description)
    inputs = {"cluster_data": cluster_data}
    result = cluster_label_crew.kickoff(inputs=inputs)  # Synchronous execution
    # Extract the label from the task output
    label_output = cluster_label_task.output
    return label_output.raw.strip()  # .raw contains the raw text output of the task
