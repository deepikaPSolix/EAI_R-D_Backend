from typing import List

import pandas as pd
from app.llm_model import LLMModel
from app.models.label_governance_model import AttributesModel


class DynamicExtractor:
    def __init__(self) -> None:
        self.model = LLMModel()

    def extract(self, files:pd.DataFrame) -> pd.DataFrame:
        extracted_attributes = [self.extract_from_file(f.chunks) for f in files.itertuples()]
        files['attributes'] = extracted_attributes
        return files

    def extract_from_file(self, chunks: list) -> AttributesModel:
        chunks_text = [c.text for c in chunks]
        res = []
        for i in range(0, len(chunks_text), 10):
            curr_chunks = chunks_text[i: min(i + 10, len(chunks_text))]
            attr_res = self.model.infer_model(self._extraction_query(curr_chunks), AttributesModel)
            if attr_res:
                res.append(attr_res)

        merged_data = self.merge_extracted_data(res)

        return merged_data

    def merge_extracted_data(self, extracted_data: List[AttributesModel]) -> AttributesModel:
        if not extracted_data:
            return AttributesModel(
                sensitivity=0,
                responsible_values=[],
                retention_time="",
                data_classifiers=[],
                attributes={}
            )
        max_sens_obj = max(extracted_data, key=lambda attr: attr.sensitivity)
        sensitivity, responsible_values = max_sens_obj.sensitivity, max_sens_obj.responsible_values

        unique_classifiers = set()
        for attr in extracted_data:
            unique_classifiers.update(attr.data_classifiers)
        
        unique_classifiers = list(unique_classifiers)

        retention_time = extracted_data[0].retention_time

        attributes_list = [a.attributes for a in extracted_data]
        attributes = self._merge_json_objects(attributes_list)

        # 🔍 NEW: Check if ANY chunk detected this as an RCC document
        rcc_obj = next((attr for attr in extracted_data if attr.is_rcc_document), None)
        
        if rcc_obj:
            # If RCC detected, preserve ALL RCC-specific fields
            return AttributesModel(
                sensitivity=sensitivity,
                responsible_values=responsible_values,
                retention_time=rcc_obj.retention_time,  # Use RCC retention time if available
                data_classifiers=unique_classifiers,
                attributes=attributes,
                is_rcc_document=True,
                record_class_code=rcc_obj.record_class_code,
                record_class_name=rcc_obj.record_class_name,
                jurisdiction=rcc_obj.jurisdiction,
                retention_event=rcc_obj.retention_event,
                record_type_examples=rcc_obj.record_type_examples,
                related_record_classes=rcc_obj.related_record_classes,
                trigger_column_patterns=rcc_obj.trigger_column_patterns
            )
        else:
            # Regular document (not RCC)
            return AttributesModel(
                sensitivity=sensitivity,
                responsible_values=responsible_values,
                retention_time=retention_time,
                data_classifiers=unique_classifiers,
                attributes=attributes
            )

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

    def _extraction_query(self, data):
        return f'''
            Task: Text data Analysis and RCC Detection

            You are provided with text from a file. Perform the following tasks accurately:

            **FIRST**: Determine if this is a Record Class Code (RCC) Definition Document
            - Check if the text contains ALL of these required fields (in ANY format - with or without colons):
              * "Record Class Code" (with or without ':') followed by or on the next line containing a code like ADM150, LEG120, etc.
              * "Record Class Name" (with or without ':') followed by or on the next line containing the name
              * "Retention Period" (with or without ':') followed by or on the next line containing a duration
            - If ALL three fields are present ANYWHERE in the document, set is_rcc_document=true and extract RCC-specific fields (see below)
            - If ANY are missing, set is_rcc_document=false and skip RCC fields
            - IMPORTANT: Be flexible with formatting - labels may be on separate lines from values, in tables, or in various layouts

            **FOR ALL DOCUMENTS** (RCC or not), extract:

            1. **Sensitivity Classification**: For the given text, assign a sensitivity level (integer) based on the following categories:
            - 1: Public Data (e.g., public reports, statistics)
            - 2: Internal Data (internal use, not for external sharing)
            - 3: Confidential Data (personal or sensitive information)
            - 4: Restricted Data (highly sensitive, access limited)
            - 5: Private Data (personal data protected by privacy laws)
            - 6: Critical Data (vital for urgent care or life-saving actions)
            - 7: Regulatory Data (compliance with legal/regulatory rules)

            Ensure text containing PII, trade secrets, legal, medical, or intellectual property are classified as level 3 or higher. Assign levels 6 or 7 for critical or regulatory data.

            2. **Responsible data types**: Analyze the text and identify 3 to 5 **distinct** and meaningful data types that justify the sensitivity classification. Do not rely on the example data types provided below—these are only for reference. The identified data types must be directly related to the actual content of the chunk. The data types should be relevant to the chunk's content, and similar types must not be repeated.

            Reference examples (for understanding only, do not use as output unless relevant):
                - PII, PHI, EHR Data, Medical History, Lab Results, Prescription Data
                - Patient Satisfaction Surveys, Appointment Records, Demographic Information
                - Contact Information, Health Insurance Details, Caregiver Information
                - Clinical Trial Data, Adverse Event Reports, Imaging Data, Genetic Information
                - Diagnosis Codes (ICD-10), Treatment Protocols, Medical Devices Information
                - Immunization Records, Clinical Notes, Anonymized Clinical Notes
                - Research Findings, Billing Information, Insurance Claims Data
                - Compliance Reports, Audit Trails, Internal Policies
                - Staff Scheduling Information, Facility Management Data
                - Equipment Inventory, Financial Reports, Strategic Plans
                - Billing Disputes, Operational Efficiency Metrics, Staffing Levels
                - Risk Management Reports

            3. **Data Points**: Provide up to 10 keys that contribute to the sensitivity level.

            4. **Retention Period**: Assign a retention period for the data in the text based on its type, specifying the time in years and months.

            5. **RCC Detection (only if applicable)**:
            - Determine if the text is a "Record Class Code (RCC) definition document".
            - If YES, set `is_rcc_document=true` and extract:
            - record_class_code (e.g., ADM150): check for RECORD CLASS CODE label
            - record_class_name :check for RECORD CLASS NAME label
            - jurisdiction (if present) :check for JURISDICTION label
            - retention_event (exact sentence if present) :check for RETENTION EVENT label
            - record_type_examples (bullet list)
            - related_record_classes (list of RCC codes mentioned)
            - trigger_column_patterns (list of column-name variations implied by the retention event; if not mentioned, leave empty)
            - If NO, set `is_rcc_document=false` and leave all RCC fields blank/empty.
        Important:
        - Do not guess RCC values.
        - Only fill RCC fields when the text clearly contains them.
        - If uncertain, leave blank.
            Return the output strictly in the required JSON format, without any additional information i.e *NO PREFIX AND SUFFIX*.

            File text:
            {data}
            '''