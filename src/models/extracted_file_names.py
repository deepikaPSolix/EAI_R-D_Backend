from pydantic import BaseModel, Field
from typing import List

class ExtractedFilesModel(BaseModel):
    files: List[str] = Field(default_factory=list, description="List of file names")
