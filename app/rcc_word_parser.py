    # app/rcc_word_parser.py
from typing import List, Optional
from pathlib import Path

from docx import Document
from flask import current_app

from app.models.rcc_models import RCCDefinition


class RCCWordParser:
    """
    Parses RCC Word documents like ADM150.docx into RCCDefinition objects.
    This is heuristic but grounded in the structure you showed:
      - Record Class Code
      - Record Class Name
      - Description
      - Record Type Examples (bullets)
      - Retention Event
      - Retention Period
      - Related Record Classes, Jurisdiction
    """


    def _defs_from_idc_attributes(self, data_rows):
        defs = []
        seen = set()

        for row in data_rows:
            attrs = row.get("attributes") or {}
            # your pipeline sometimes stores model_dump already; ensure dict
            if isinstance(attrs, str):
                # if ever string, skip here (we’ll fix evaluation separately)
                continue

            if attrs.get("is_rcc_document") is True and attrs.get("record_class_code"):
                code = attrs.get("record_class_code")
                if code in seen:
                    continue
                seen.add(code)

                defs.append(RCCDefinition(
                    code=code,
                    name=attrs.get("record_class_name"),
                    description=(attrs.get("attributes") or {}).get("Description"),
                    record_type_examples=attrs.get("record_type_examples") or [],
                    jurisdiction=attrs.get("jurisdiction"),
                    related_record_classes=attrs.get("related_record_classes") or [],
                    retention_event=attrs.get("retention_event"),
                    retention_period_years=self._parse_years(attrs.get("retention_time")),
                    retention_period_months=None,
                    keywords=[],
                    table_name_examples=[],
                    trigger_column_name_patterns=attrs.get("trigger_column_patterns") or [],
                ))
        return defs

    def _parse_years(self, retention_time):
        # retention_time might be "1 Year" or "5 years"
        if not retention_time:
            return None
        import re
        m = re.search(r"(\d+)", str(retention_time))
        return int(m.group(1)) if m else None

    def parse_file(self, file_path: str) -> Optional[RCCDefinition]:
        try:
            doc = Document(file_path)
        except Exception as e:
            current_app.logger.error(f"RCCWordParser: failed to open {file_path}: {e}")
            return None

        # Extract data from tables (common format: label | value)
        table_data = {}
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if len(cells) >= 2:
                    label = cells[0].strip().lower()
                    value = cells[1].strip()
                    if label and value:
                        table_data[label] = value

        # Also collect all text lines from paragraphs (fallback/additional data)
        lines: List[str] = []
        for para in doc.paragraphs:
            text = (para.text or "").strip()
            if text:
                lines.append(text)

        # Enhanced find_value: check table data first, then paragraph lines
        def find_value(prefixes: List[str]) -> Optional[str]:
            # 1. Check table data (most reliable for structured docs)
            for p in prefixes:
                if p.lower() in table_data:
                    return table_data[p.lower()]
            
            # 2. Check paragraph lines (fallback)
            for line in lines:
                low = line.lower()
                for p in prefixes:
                    if low.startswith(p.lower()):
                        # take text after colon if present
                        parts = line.split(":", 1)
                        return parts[1].strip() if len(parts) == 2 else line.strip()
            return None

        # Enhanced description extraction: handle heading + content paragraphs
        def find_section_content(heading_prefixes: List[str]) -> Optional[str]:
            """Extract content that follows a heading like 'Description'."""
            # First try table data
            for p in heading_prefixes:
                if p.lower() in table_data:
                    return table_data[p.lower()]
            
            # Then try finding heading and collecting following paragraphs
            try:
                idx = next(
                    i for i, line in enumerate(lines)
                    if any(line.lower().startswith(prefix.lower()) for prefix in heading_prefixes)
                )
                
                # Check if heading has content after colon on same line
                if ":" in lines[idx]:
                    parts = lines[idx].split(":", 1)
                    if parts[1].strip():
                        return parts[1].strip()
                
                # Otherwise, gather content from following paragraphs until next heading
                content_lines = []
                stop_headings = [
                    "record type examples", "retention event", "retention period",
                    "jurisdiction", "related record classes", "united states event date"
                ]
                
                for line in lines[idx + 1:]:
                    low = line.lower()
                    # Stop if we hit another known heading
                    if any(low.startswith(sh) for sh in stop_headings):
                        break
                    # Stop if line looks like a heading (all caps, short, ends with colon)
                    if line.isupper() and len(line) < 50:
                        break
                    if line.strip() and not line.endswith(":"):
                        content_lines.append(line.strip())
                
                if content_lines:
                    return " ".join(content_lines)
            except StopIteration:
                pass
            
            return None

        code = find_value(["record class code"])
        name = find_value(["record class name"])
        description = find_section_content(["description", "overview", "definition", "purpose"])

        # Record Type Examples: gather all bullet-like lines after the heading
        record_type_examples: List[str] = []
        try:
            # find index of heading
            idx = next(
                i for i, line in enumerate(lines)
                if line.lower().startswith("record type examples")
            )
            for line in lines[idx + 1:]:
                if line.lower().startswith("united states event date") or \
                   line.lower().startswith("retention event") or \
                   line.lower().startswith("related record classes"):
                    break
                # treat any non-empty line as example (bullets often start with • or -)
                if line.strip():
                    record_type_examples.append(line.strip("•- ").strip())
        except StopIteration:
            pass

        retention_event = find_value(["retention event"])
        retention_period_raw = find_value(["retention period"])
        retention_years: Optional[int] = None
        retention_months: Optional[int] = None
        if retention_period_raw:
            low = retention_period_raw.lower()
            # Very simple parse: look for "<n> year" and "<m> month"
            import re
            year_match = re.search(r"(\d+)\s+year", low)
            month_match = re.search(r"(\d+)\s+month", low)
            if year_match:
                retention_years = int(year_match.group(1))
            if month_match:
                retention_months = int(month_match.group(1))

        jurisdiction = find_value(["jurisdiction"])
        related_raw = find_value(["related record classes"])
        related_classes: List[str] = []
        if related_raw:
            # e.g. "See ADM100 for Administration; See INV160 for ..."
            import re
            related_classes = re.findall(r"(ADM\d+|LEG\d+|INV\d+)", related_raw)

        # derive simple keywords from name + description + examples
        keywords_source = " ".join(
            [name or "", description or ""] + record_type_examples
        )
        keywords = list({
            w.strip(" ,.;()").lower()
            for w in keywords_source.split()
            if len(w) > 3
        })

        if not code:
            current_app.logger.warning(
                f"RCCWordParser: no Record Class Code found in {file_path}"
            )
            return None

        definition = RCCDefinition(
            code=code.strip(),
            name=name,
            description=description,
            record_type_examples=record_type_examples,
            jurisdiction=jurisdiction,
            related_record_classes=related_classes,
            retention_event=retention_event,
            retention_period_years=retention_years,
            retention_period_months=retention_months,
            keywords=keywords,
            table_name_examples=[],             # will fill from Excel later if needed
            trigger_column_name_patterns=[],    # will infer from schema/data later
        )

        current_app.logger.info(
            f"RCCWordParser: parsed RCC definition {definition.code} from {Path(file_path).name}"
        )
        return definition
