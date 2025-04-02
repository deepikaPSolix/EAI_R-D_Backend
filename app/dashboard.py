from flask import current_app
from lida import Manager, TextGenerationConfig , llm
from app.llm_model import LLMModel
from app.chroma_db import ChromaDB
from pathlib import Path

from app.rag import RAG
from io import StringIO
import tempfile
import os
import uuid
import json
import pandas as pd


class Dashboard:
    def generate_csv_from_response(self, response, model_source="together"):
       
        # Use LLM to convert response into CSV format
        llm_model = LLMModel(model_source).model
        prompt = f"""
        You are a data analyst working with an AI visualization system.

        From the following text, extract all relevant tabular data and convert it into a **strict, valid CSV format**.

        ### Instructions:
        - Output **must only be raw CSV** — do NOT wrap it in markdown (no ```csv or ```)
        - This CSV will be parsed directly by a chart generation system (LIDA), so **any formatting error will break it**
        - Use appropriate column names based on the data.
        - Only return the CSV (no extra text or commentary).

        ### Text:
        {response}
        """
        llm_csv_response = llm_model.invoke(prompt).content.strip()

        # Save CSV to disk
        file_name = f"{uuid.uuid4().hex}.csv"
        output_dir = "cache"
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, file_name)

        with open(file_path, "w") as f:
            f.write(llm_csv_response)

        # If only one row, duplicate it for LIDA stability
        df = pd.read_csv(file_path)
        if len(df) == 1:
            df = pd.concat([df, df.copy()], ignore_index=True)
            df.to_csv(file_path, index=False)
        

        current_app.logger.info(f"{file_path}")

        return file_path
    

    def run_lida_on_csv(self, csv_path, user_query, api_key):
        """
        Takes path to CSV file and user query,
        runs LIDA summarization, goal generation, and visualization.
        Returns generated charts.
        """
        # Load data
        df = pd.read_csv(csv_path)

        # Fill missing values in numeric columns
        for column in df.select_dtypes(include=['number']).columns:
            df[column] = df[column].fillna(0)

        # Initialize LIDA
        lida = Manager(text_gen=llm("openai", api_key=api_key))
        textgen_config = TextGenerationConfig(n=1, temperature=0.5, model="gpt-4o", use_cache=True)

        # Generate summary and goals
        summary = lida.summarize(df, summary_method="default", textgen_config=textgen_config)
        goals = lida.goals(summary, n=2, textgen_config=textgen_config)

        current_app.logger.info("\nGenerated Goals:")
        
        # Visualize data for the original user query
        viz_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
        charts = lida.visualize(summary=summary, goal=user_query, textgen_config=viz_config)

        chart = charts[0] 

        # --- Save chart image using built-in method
        file_name = f"{uuid.uuid4().hex}.png"
        output_dir = Path("cache/images")
        output_dir.mkdir(exist_ok=True)
        image_path = output_dir / file_name

        try:
            chart.savefig(str(image_path))  # 👈 This uses .raster and decodes it
            current_app.logger.info(f"LIDA chart saved to: {image_path}")
            return str(image_path)
        except Exception as e:
            current_app.logger.error(f"Failed to save chart image: {e}")
            return "Failed to generate the image"
            raise e
        
        