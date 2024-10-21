from pydantic import BaseModel, Field

class LabelGovernanceModel(BaseModel):
    label: str = Field(default="", description="The label for the data in the given cluster.")
    sensitivity: int = Field(default=0, description="The sensitivity level of the document")
    retention_time: str = Field(default = "", description="The time for which this data should be retained based on the type of data.")