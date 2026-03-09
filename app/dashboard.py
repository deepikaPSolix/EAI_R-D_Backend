from flask import current_app
from lida import Manager, TextGenerationConfig , llm
from lida.utils import plot_raster
from app.llm_model import LLMModel
from app.chroma_db import ChromaDB
from pathlib import Path
from app.rag import RAG
from io import StringIO
import tempfile
import os
import uuid
import json
import re
import subprocess
from pathlib import Path
from glob import glob
import pandas as pd
import huggingface_hub as hf
hf.cached_download = hf.hf_hub_download 

class Dashboard:
    def generate_csv_from_response(self, response, model_source="openai"):
       
        # Use LLM to convert response into CSV format
        llm_model = LLMModel(model_source).model
        prompt = f"""
            You are a data analyst working with an AI visualization system.

            From the following text, extract all relevant tabular data and convert it into a **strict, valid CSV format** that is optimized for visual analysis.

            ### Requirements:
            - Output **only raw CSV** (no markdown or code block formatting).
            - Use consistent column headers — every row must contain the exact same number of fields.
            - Avoid placing compound values into a single cell.
            - Flatten nested data — use one row per record.
            - Aggregate values by time or category **only if** it adds clarity (e.g. total monthly expenses).
            - If any value like "Total Cost" appears separately, skip it unless it's part of a structured table.
            - Always include a clear, sortable column like `Date` or `Category` if applicable.
            - Keep the data clean and minimal — **don't include metadata or notes**.

        ### Text:
        {response}
        """
        llm_csv_response = llm_model.invoke(prompt).content.strip()

        # Save CSV to disk
        file_name = f"{uuid.uuid4().hex}.csv"
        output_dir = Path("cache/csv")
        output_dir.mkdir(exist_ok=True)
        file_path = output_dir / file_name
      
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
            return "Failed to generate the image + " + str(e)
           

        # #generate 3D visulaization of the graphs/dashboards
        # current_app.logger.info("Calling Lida infographics for 3D visualization")

        # import time
        # t0 = time.time()
        # try:
        #     infographics = lida.infographics(visualization =chart.raster, n=1, style_prompt="generate any 3D graph choose the best colors and pattern ")    
        #     current_app.logger.info("LIDA infographics generated successfully")
        #     current_app.logger.info(f'type(plot_raster([chart.raster, infographics["images"][0]]))')
            
        # except Exception as e:
        #     current_app.logger.error(f"Error generating infographics: {e}")
        #     raise
        # t1 = time.time()
        # current_app.logger.info(f"Infographics generation time: {t1 - t0} seconds")
        


    def generate_dashboard(self, response):
        # Define output folders
        current_app.logger.info("Generating dashboard... with the llm")
        code_dir = Path("cache/code")
        image_dir = Path("cache/images")
        code_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

        # Extract Python code block from LLM response
        pattern = r"```python(.*?)```"
        match = re.search(pattern, response, re.DOTALL)
        if not match:
            print("⚠️ No Python code block found.")
            current_app.logger.info("⚠️ No Python code block found.")
            return None

        python_code = match.group(1).strip()
        current_app.logger.info(f"Extracted Python code:\n{python_code}")

        # Create unique filenames
        uid = uuid.uuid4().hex
        code_path = code_dir / f"{uid}.py"
        image_path = image_dir / f"{uid}.png"

        # Append plt.savefig if not included
        if "plt.savefig" not in python_code:
            python_code += f"\nimport matplotlib.pyplot as plt\nplt.savefig('{image_path.as_posix()}')\n"

        # Write code to file
        with open(code_path, "w") as f:
            f.write(python_code)
        current_app.logger.info(f"✅Code saved to: {code_path}")
        # Execute the code
        try:
            subprocess.run(["python", str(code_path)], check=True)
            print(f"✅ Code executed: {code_path}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Code execution failed: {e}")
            return None

        if image_path.exists():
            current_app.logger.info(f"✅ Chart image saved to: {image_path}")
        else:
            current_app.logger.info(f"⚠️ No chart image was created.")
            return None

        
        return str(image_path)