# models/rcc_models.py
from typing import List, Optional, Dict
from pydantic import BaseModel


class RCCDefinition(BaseModel):
    code: str
    name: Optional[str] = None
    description: Optional[str] = None

    record_type_examples: List[str] = []

    jurisdiction: Optional[str] = None
    related_record_classes: List[str] = []

    retention_event: Optional[str] = None
    retention_period_years: Optional[int] = None
    retention_period_months: Optional[int] = None

    keywords: List[str] = []
    table_name_examples: List[str] = []
    trigger_column_name_patterns: List[str] = []


class TableColumnProfile(BaseModel):
    name: str
    data_type: Optional[str] = None
    is_nullable: Optional[bool] = None

    ordinal_position: Optional[int] = None
    column_default: Optional[str] = None
    char_max_length: Optional[int] = None
    numeric_precision: Optional[int] = None
    numeric_scale: Optional[int] = None
    date_time_precision: Optional[int] = None
    sample_values: List[str] = []


class TableProfile(BaseModel):
    file_name: str
    table_catalog: Optional[str] = None
    table_schema: Optional[str] = None
    table_name: str
    source_sheet_name: Optional[str] = None

    columns: List[TableColumnProfile] = []
    row_count: int = 0
    sample_rows: List[Dict[str, str]] = []


class IDCAttributes(BaseModel):
    sensitivity: Optional[int] = None
    responsible_values: List[str] = []
    retention_time: Optional[str] = None
    data_classifiers: List[str] = []
    attributes: Dict[str, str] = {}


class RCCCandidate(BaseModel):
    code: str
    score: float = 0.0

    # NEW
    confidence: Optional[float] = None  # 0..1 (UI can show %)

    selected: bool = False

    reason: str = ""
    not_selected_reason: Optional[str] = None

    trigger_column: Optional[str] = None
    trigger_reason: Optional[str] = None

    retention_event: Optional[str] = None
    retention_period_years: Optional[int] = None
    retention_period_months: Optional[int] = None


class RCCClassificationResult(BaseModel):
    file_name: str
    table_name: str

    best_code: Optional[str] = None
    candidates: List[RCCCandidate] = []
