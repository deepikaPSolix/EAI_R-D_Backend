from pydantic import BaseModel, Field
from typing import Dict, List, Any

class GovernanceAttributes(BaseModel):
    label: str = Field(default="", description="The specific label representing the content of the document cluster.")
    sensitivity: int = Field(default="", description="The assigned sensitivity level of the document based on classification rules (1 to 7).")
    data_classifiers: str = Field(default="", description="List atleast 1 and atmost 5 distinct and meaningful types of data present in the document, returned in CSV format. Do not list basic types like Name, age, address ,etc but combine them into a single type like PII.")
    responsible_values: str = Field(default="", description="Up to 10 keys representing the data points that justify the sensitivity level, formatted as name1, name2.")
    file_name: str = Field(default="", description="The name of the document file.")
    retention_time: str = Field(default="", description="The retention period for this document, specified in years and months based on the document type.")

class GovernanceModel(BaseModel):
    attributes: List[GovernanceAttributes] = Field(description="The governance attributes of all the documents")