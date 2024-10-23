from pydantic import BaseModel, Field
from typing import Dict, List, Any

class GovernanceAttributes(BaseModel):
    label: str = Field(default="", description="The label for the data in the given cluster.")
    sensitivity: int = Field(default=0, description="The sensitivity level of the document")
    reason: str = Field(default="", description="Sensitivity attributes i.e. The data points responsible for the sensitivity value. The value must be a string. For example 'PII, Name, Medical Data'")
    file_name: str = Field(default="", description="File name of the document.")
    retention_time: str = Field(default = "", description="The time for which this data should be retained based on the type of data.")

class GovernanceModel(BaseModel):
    attributes: List[GovernanceAttributes] = Field(description="The governance attributes of all the documents")