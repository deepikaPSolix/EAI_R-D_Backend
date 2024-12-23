from pydantic import BaseModel, ConfigDict
from typing import Optional

from app.models.label_governance_model import AttributesModel

class DocFile(BaseModel):
    file_name: str
    file_type: str
    chunks: list
    data: str
    cluster: Optional[int] = None
    cluster_label: Optional[str] = None
    attributes: Optional[AttributesModel] = None

    # Config
    model_config = ConfigDict(frozen=False)
