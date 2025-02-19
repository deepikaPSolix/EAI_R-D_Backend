from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Dict, List, Any

class GovernanceAttributes(BaseModel):
    label: str = Field(default="", description="The specific label representing the content of the document cluster.")
    sensitivity: int = Field(default="", description="The assigned sensitivity level of the document based on classification rules (1 to 7).")
    data_classifiers: str = Field(default="", description="List atleast 1 and atmost 5 distinct and meaningful types of data present in the document, returned in CSV format. Do not list basic types like Name, age, address ,etc but combine them into a single type like PII.")
    responsible_values: str = Field(default="", description="Up to 10 keys representing the data points that justify the sensitivity level, formatted as name1, name2.")
    file_name: str = Field(default="", description="The name of the document file.")
    retention_time: str = Field(default="", description="The retention period for this document, specified in years and months based on the document type.")

class GovernanceModel(BaseModel):
    label: str = Field(default="", description="The specific label representing the content of the document cluster.")

class AttributesModel(BaseModel):
    sensitivity: int = Field(0, description="The assigned sensitivity level of the document based on classification rules (1 to 7).")
    responsible_values: List[str] = Field(default_factory=list, description="Up to 10 keys representing the data points that justify the sensitivity level, formatted as a list of strings.")
    data_classifiers: List[str] = Field(default_factory=list, description="List of 1 to 5 distinct, meaningful data types present in the document (e.g., PII, EHR), excluding basic types like name, age, address.")
    retention_time: str = Field("", description="The retention period for this document, specified in years and months based on the document type.")
    attributes: Dict[Any, Any] = Field(default_factory=dict, description="Relevant data and attributes from the given text as key-value pairs.")

    model_config = ConfigDict(frozen=False, validate_default = False, extra = "allow")

    @field_validator("sensitivity", mode="before")
    def set_default_location(cls, value):
        if value not in range(1, 8):
            return 0
        return value