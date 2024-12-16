from pydantic import BaseModel

class DocFile(BaseModel):
    file_name: str
    file_type: str
    chunks: list
    data: str