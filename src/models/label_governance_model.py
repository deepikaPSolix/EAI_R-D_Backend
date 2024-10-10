from pydantic import BaseModel, Field

class LabelGovernanceModel(BaseModel):
    label: str = Field(default="", description="The label for the data in the given cluster.")
    sensitivity: str = Field(default="0", description="The sensitivity level of the document")