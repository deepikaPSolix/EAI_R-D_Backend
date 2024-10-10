from doc_file import DocFile
from llm_model import LLMModel
from models.medical_data_model import ExtractedData
from typing import List
import pandas as pd


class AttributeExtractor:
    def __init__(self):
        self.model = LLMModel()

    def extract_from_files(self, files: List[DocFile]):
        res = []
        for file in files:
            data = self.extract_data_from_chunks(file.chunks)
            combined_data = self.combine_data(data, file.file_name)
            res.append(combined_data)
        return pd.DataFrame(res)

    def extract_data_from_chunks(self, chunks):
        res = []

        for i in range(0, len(chunks), 10):
            query = '''
            Extract attributes described in the output format from the text below.
            - If you can't find an attribute, just leave it blank. Do not put null.
            - If there are multiple values for the same attribute, select the most relevant one based on the context provided.
            - Make sure to check is the text is medical data.
            - Make sure the output data complies with the output format.
            - Only output the JSON and absolutely nothing else.
            Here is the text:
            '''

            for j in range(i, min(i + 10, len(chunks))):
                query += chunks[j].text + "\n"

            attr_res = self.model.infer_model(query, ExtractedData)
            if attr_res:
                res.append(attr_res)

        return res

    def combine_data(self, data, file_name):
        rows = [item.dict() for item in data]
        combined_data = self._merge_json_objects(rows)
        combined_data['file_name'] = file_name
        return combined_data

    def _merge_json_objects(self, json_objects: dict):
        # Initialize an empty dictionary to hold the final merged JSON object
        merged_data = {}
        
        # Iterate over each JSON object to merge
        for json_obj in json_objects:
            for key, value in json_obj.items():
                # Handle if the key is already present in the merged_data
                if key in merged_data:
                    existing_value = merged_data[key]

                    # If the value is a list, convert both the existing value and new value to strings and merge
                    if isinstance(value, list):
                        if not value:  # Handle empty arrays
                            merged_data[key] = ""
                        else:
                            # Convert both existing and new values to strings
                            existing_value_str = ', '.join(map(str, existing_value)) if isinstance(existing_value, list) else str(existing_value)
                            value_str = ', '.join(map(str, value))
                            merged_data[key] = existing_value_str + ', ' + value_str

                    # If the value is boolean, set to True if any are True
                    elif isinstance(value, bool):
                        merged_data[key] = merged_data[key] or value

                    # Otherwise, replace with the most relevant value
                    else:
                        merged_data[key] = value
                else:
                    # If key does not exist, simply add it
                    if isinstance(value, list):
                        merged_data[key] = ', '.join(map(str, value))  # Convert list to string immediately
                    else:
                        merged_data[key] = value

        return merged_data