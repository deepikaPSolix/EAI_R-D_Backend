# app/rcc_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import os
import json
from pathlib import Path

import pandas as pd
from flask import current_app

from app.rcc_word_parser import RCCWordParser
from app.rcc_explainer import RCCExplainerLLM
from app.models.rcc_models import (
    TableProfile,
    TableColumnProfile,
    RCCClassificationResult,
    RCCCandidate,
    RCCDefinition,
)


class RCCEngine:
    """
    RCC Engine (Deterministic core + optional LLM explanation layer)

    What this does today:
    - Parse RCC definitions (from Word docs if possible; otherwise from IDC extracted attributes)
    - Profile Excel schema into TableProfile list
    - Deterministically score RCC codes for each table using:
        * table/schema name tokens vs record_type_examples / keywords
        * column name tokens vs keywords
        * retention-event-driven trigger patterns from RCC definition
        * presence of datetime columns if retention is date-driven
    - Pick best code + top alternatives with:
        * reason (why selected)
        * not_selected_reason (why not)
        * confidence % (deterministic)
        * trigger_column + trigger_reason (deterministic)

    Optional:
    - If RCC_USE_LLM_EXPLANATIONS=true, call LLM ONLY to rephrase the explanation.
      LLM never decides code or trigger. If LLM output invalid -> ignore safely.
    """

    # -----------------------------
    # Public entry
    # -----------------------------
    def attach_rcc_results(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(data, list):
            data = [data]

        current_app.logger.info(
            f"RCCEngine.attach_rcc_results called for {len(data)} file(s) [deterministic+optional-LLM]"
        )

        # Ensure stable keys exist
        for row in data:
            row.setdefault("rcc_result", None)
            row.setdefault("file_path", None)
            row.setdefault("attributes", {})

        # 1) Collect RCC definitions (Word parse + IDC fallback)
        rcc_definitions: List[RCCDefinition] = []

        # 1a) Try parsing real Word RCC docs (best source)
        parser = RCCWordParser()
        for row in data:
            if not self._is_docx(row):
                continue
            path = row.get("file_path")
            if not path or not os.path.isfile(path):
                continue
            # Use your existing heuristic OR accept explicitly flagged is_rcc_document
            if not self._looks_like_rcc_doc(row):
                continue

            try:
                definition = parser.parse_file(path)
                if definition and definition.code:
                    rcc_definitions.append(definition)
            except Exception as e:
                current_app.logger.warning(f"RCCWordParser failed for {path}: {e}")

        # 1b) ALSO build defs from IDC extracted attributes (your current logs show these fields)
        #     This fixes your earlier crash: _defs_from_idc_attributes missing.
        rcc_definitions.extend(self._defs_from_idc_attributes(data))

        # De-dup by code with SMART MERGE (preserve Word doc data when richer)
        defs_by_code: Dict[str, RCCDefinition] = {}
        for d in rcc_definitions:
            if not d.code:
                continue
            code_upper = d.code.upper()
            
            if code_upper in defs_by_code:
                # Merge: keep Word document fields if they're richer
                existing = defs_by_code[code_upper]
                
                # Prefer existing (Word doc) description if present, otherwise take new
                d.description = existing.description or d.description
                
                # Merge lists (unique values)
                if existing.record_type_examples:
                    d.record_type_examples = list(dict.fromkeys(
                        existing.record_type_examples + (d.record_type_examples or [])
                    ))
                if existing.related_record_classes:
                    d.related_record_classes = list(dict.fromkeys(
                        existing.related_record_classes + (d.related_record_classes or [])
                    ))
                if existing.trigger_column_name_patterns:
                    d.trigger_column_name_patterns = list(dict.fromkeys(
                        existing.trigger_column_name_patterns + (d.trigger_column_name_patterns or [])
                    ))
                
                # Keep other Word doc fields if present
                d.name = existing.name or d.name
                d.jurisdiction = existing.jurisdiction or d.jurisdiction
                d.retention_event = existing.retention_event or d.retention_event
                d.retention_period_years = existing.retention_period_years if existing.retention_period_years is not None else d.retention_period_years
                d.retention_period_months = existing.retention_period_months if existing.retention_period_months is not None else d.retention_period_months
            
            defs_by_code[code_upper] = d
        rcc_definitions = list(defs_by_code.values())

        if rcc_definitions:
            self._save_definitions_to_cache(rcc_definitions)
            current_app.logger.info(f"RCCEngine: loaded {len(rcc_definitions)} RCC definition(s).")
            for _d in rcc_definitions:
                current_app.logger.info(
                    f"[RCC-DEF] code={_d.code} | name={_d.name} | "
                    f"description_len={len(_d.description or '')} | "
                    f"description_snippet='{(_d.description or '')[:120]}' | "
                    f"examples={_d.record_type_examples[:5]} | "
                    f"keywords_count={len(_d.keywords)} | "
                    f"keywords_sample={_d.keywords[:15]} | "
                    f"retention_event={_d.retention_event} | "
                    f"trigger_patterns={_d.trigger_column_name_patterns}"
                )
        else:
            current_app.logger.info("RCCEngine: no RCC definitions found in this batch.")

        # 2) Profile Excel tables (schema)
        excel_rows = [row for row in data if self._is_excel(row)]
        all_profiles: List[TableProfile] = []
        for row in excel_rows:
            path = row.get("file_path")
            if not path or not os.path.isfile(path):
                current_app.logger.warning(
                    f"RCCEngine: missing file_path for Excel {row.get('file_name')}"
                )
                continue

            try:
                profiles = self._profile_excel_schema(path, row.get("file_name") or Path(path).name)
                if profiles:
                    all_profiles.extend(profiles)
                    current_app.logger.info(
                        f"RCCEngine: profiled {len(profiles)} tables from Excel {row.get('file_name')}"
                    )
            except Exception as e:
                current_app.logger.error(f"RCCEngine: error profiling Excel {row.get('file_name')}: {e}")

        if all_profiles:
            self._save_profiles_to_cache(all_profiles)
        else:
            current_app.logger.info("RCCEngine: no table profiles extracted from Excel(s).")

        # 3) Deterministic classification for EACH table profile
        if rcc_definitions and all_profiles:
            results_by_excel: Dict[str, List[RCCClassificationResult]] = self._classify_profiles(
                all_profiles, rcc_definitions
            )

            # Attach into each Excel row in 'data' (by filename)
            for row in data:
                if not self._is_excel(row):
                    continue
                fname = (row.get("file_name") or "").lower()
                if fname in results_by_excel:
                    row["rcc_result"] = [r.model_dump() for r in results_by_excel[fname]]
                    current_app.logger.info(
                        f"RCCEngine: attached {len(results_by_excel[fname])} table RCC result(s) to {row.get('file_name')}"
                    )

        return data

    # -----------------------------
    # Detect file types
    # -----------------------------
    def _is_docx(self, row: Dict[str, Any]) -> bool:
        return (row.get("file_type") or "").lower() == "docx"

    def _is_excel(self, row: Dict[str, Any]) -> bool:
        return (row.get("file_type") or "").lower() in ("xlsx", "xls")

    def _looks_like_rcc_doc(self, row: Dict[str, Any]) -> bool:
        # Accept explicit flag from IDC if present
        attrs = row.get("attributes") or {}
        if isinstance(attrs, dict) and attrs.get("is_rcc_document") is True:
            return True

        # Fallback heuristic: doc text contains markers
        data_text = (row.get("data") or "").lower()
        return ("record class code" in data_text) or ("retention event" in data_text)

    def _extract_description_from_raw_text(self, row: Dict[str, Any]) -> Optional[str]:
        """
        Fallback description extraction from raw document text.
        Looks for 'Description:' or similar patterns in the document.
        """
        raw_text = row.get("data") or ""
        if not raw_text:
            return None
        
        # Try to find description section
        lines = raw_text.split("\n")
        for i, line in enumerate(lines):
            line_lower = line.strip().lower()
            
            # Look for "Description:" heading
            if line_lower.startswith("description"):
                # Check if description is on same line after colon
                if ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) == 2 and parts[1].strip():
                        return parts[1].strip()
                
                # Otherwise, gather next few lines until we hit another heading
                content_lines = []
                stop_headings = [
                    "record type examples", "retention event", "retention period",
                    "jurisdiction", "related record classes", "united states"
                ]
                
                for next_line in lines[i + 1:]:
                    next_lower = next_line.strip().lower()
                    # Stop at next heading
                    if any(next_lower.startswith(sh) for sh in stop_headings):
                        break
                    # Stop at empty line followed by heading-like text
                    if not next_line.strip():
                        continue
                    # Collect content
                    if next_line.strip():
                        content_lines.append(next_line.strip())
                    # Limit to 5 lines to avoid grabbing too much
                    if len(content_lines) >= 5:
                        break
                
                if content_lines:
                    return " ".join(content_lines)
        
        return None

    # -----------------------------
    # Parse RCC defs from IDC output
    # -----------------------------
    def _defs_from_idc_attributes(self, data_rows: List[Dict[str, Any]]) -> List[RCCDefinition]:
        """
        Your DynamicExtractor already returns fields like:
          is_rcc_document, record_class_code, record_class_name, retention_event,
          record_type_examples, related_record_classes, trigger_column_patterns, jurisdiction, etc.

        We convert those into RCCDefinition objects so engine works even if Word parser fails.
        
        ENHANCED: If description is missing from IDC, extract it from raw document text.
        """
        defs: Dict[str, RCCDefinition] = {}
        for row in data_rows:
            attrs = row.get("attributes")
            if not isinstance(attrs, dict):
                continue
            if attrs.get("is_rcc_document") is not True:
                continue

            code = (attrs.get("record_class_code") or "").strip()
            if not code:
                continue

            d = defs.get(code.upper())
            if d is None:
                d = RCCDefinition(code=code.upper())

            # Fill fields (prefer not overwriting if already present)
            d.name = d.name or attrs.get("record_class_name") or None
            d.jurisdiction = d.jurisdiction or attrs.get("jurisdiction") or None
            d.retention_event = d.retention_event or attrs.get("retention_event") or None

            # description can come from attributes.Description if present
            desc = None
            if isinstance(attrs.get("attributes"), dict):
                desc = attrs["attributes"].get("Description") or attrs["attributes"].get("description")
            
            # FALLBACK: If IDC didn't extract description, try to extract from raw document text
            if not desc:
                desc = self._extract_description_from_raw_text(row)
            
            d.description = d.description or desc

            rtex = attrs.get("record_type_examples") or []
            if isinstance(rtex, list):
                # merge unique
                merged = list(dict.fromkeys((d.record_type_examples or []) + rtex))
                d.record_type_examples = merged

            rel = attrs.get("related_record_classes") or []
            if isinstance(rel, list):
                d.related_record_classes = list(dict.fromkeys((d.related_record_classes or []) + rel))

            trig = attrs.get("trigger_column_patterns") or attrs.get("trigger_column_name_patterns") or []
            if isinstance(trig, list):
                d.trigger_column_name_patterns = list(
                    dict.fromkeys((d.trigger_column_name_patterns or []) + trig)
                )

            # retention_time may exist like "1 Year" or "5 years"
            retention_time = attrs.get("retention_time")
            yrs, mos = self._parse_retention_time(retention_time)
            if d.retention_period_years is None and yrs is not None:
                d.retention_period_years = yrs
            if d.retention_period_months is None and mos is not None:
                d.retention_period_months = mos

            # keywords derived later
            defs[code.upper()] = d

        # derive keywords and default trigger patterns if missing
        out: List[RCCDefinition] = []
        for d in defs.values():
            d.keywords = self._derive_keywords(d)
            # If trigger patterns empty but retention_event suggests creation-date driven,
            # we can add generic *soft defaults* (NOT hardcoded per code — driven by event type).
            if not d.trigger_column_name_patterns:
                d.trigger_column_name_patterns = self._default_trigger_patterns_from_event(d.retention_event)
            out.append(d)
        return out

    def _parse_retention_time(self, retention_time: Any) -> Tuple[Optional[int], Optional[int]]:
        """
        Converts '1 Year', '5 years', '18 months' into (years, months).
        Conservative parse.
        """
        if not retention_time:
            return None, None
        s = str(retention_time).strip().lower()

        # Extract numbers
        import re
        m = re.search(r"(\d+)", s)
        if not m:
            return None, None
        n = int(m.group(1))

        if "year" in s:
            return n, 0
        if "month" in s:
            return 0, n
        return None, None

    def _derive_keywords(self, d: RCCDefinition) -> List[str]:
        """
        Enhanced keyword derivation focusing on Description field.
        Ensures ALL record_type_examples are included.
        Applies comprehensive stopword filtering and domain-specific term boosting.
        """
        # GUARANTEE: ALL record_type_examples tokens are included (highest priority)
        example_tokens = []
        if d.record_type_examples:
            for ex in d.record_type_examples:
                example_tokens.extend(self._tokenize(ex))
        
        # Then add Description and Name
        text_parts: List[str] = []
        if d.description:
            text_parts.append(d.description)
        if d.name:
            text_parts.append(d.name)

        desc_tokens = self._tokenize(" ".join(text_parts))
        
        # Comprehensive stopword list (common English words that add no semantic value)
        stop = {
            "the", "and", "of", "to", "a", "in", "for", "on", "with", "is", "are", 
            "this", "that", "as", "be", "by", "an", "or", "from", "at", "which", 
            "but", "not", "have", "has", "had", "do", "does", "did", "will", "would",
            "should", "could", "may", "might", "can", "must", "shall", "their", "them",
            "they", "his", "her", "its", "our", "your", "all", "each", "every", "some",
            "any", "many", "much", "more", "most", "other", "such", "only", "own",
            "same", "so", "than", "too", "very", "just", "where", "when", "who", "what",
            "why", "how", "there", "here", "then", "now", "been", "being", "were", "was",
            "am", "these", "those", "about", "into", "through", "during", "before",
            "after", "above", "below", "between", "under", "again", "further", "once",
            "related", "including", "such", "etc", "also", "well", "used"
        }
        
        # Filter example tokens (MUST be included, even if common)
        filtered_examples = [t for t in example_tokens if t not in stop and len(t) >= 3]
        
        # Filter description/name tokens
        filtered_desc = [t for t in desc_tokens if t not in stop and len(t) >= 3]
        
        # Domain-specific term boosting (these appear multiple times if found)
        domain_terms = {
            "employee", "termination", "hiring", "compensation", "performance",
            "customer", "vendor", "contract", "invoice", "payment", "transaction",
            "policy", "claim", "beneficiary", "premium", "coverage", "enrollment",
            "patient", "medical", "diagnosis", "treatment", "prescription", "clinical",
            "student", "course", "grade", "enrollment", "transcript", "degree",
            "project", "budget", "expense", "revenue", "forecast", "audit",
            "compliance", "regulatory", "legal", "litigation", "settlement",
            "intellectual", "property", "patent", "trademark", "copyright",
            "security", "incident", "breach", "access", "authentication",
            "maintenance", "repair", "asset", "equipment", "facility"
        }
        
        # Combine: Examples FIRST (guaranteed), then description
        all_tokens = filtered_examples + filtered_desc
        
        # Boost domain terms by duplicating them (increases weight in matching)
        boosted_tokens = []
        for t in all_tokens:
            boosted_tokens.append(t)
            if t in domain_terms:
                boosted_tokens.append(t)  # Add twice for higher weight
        
        # Return up to 120 keywords (increased to ensure all examples + rich description)
        # Keep order but allow duplicates for boosting effect
        return boosted_tokens[:120]

    def _default_trigger_patterns_from_event(self, retention_event: Optional[str]) -> List[str]:
        """
        NOT hardcoded by RCC code. Derived by event semantics.
        If event mentions create/created/creation -> add create-like patterns.
        If event mentions terminate/termination -> add termination-like patterns.
        etc.
        """
        if not retention_event:
            return []

        ev = retention_event.lower()
        patterns: List[str] = []

        if "create" in ev or "created" in ev or "creation" in ev:
            patterns += ["CreatedDate", "CreateDate", "InsertedDate", "InsertDate", "DateCreated"]
        if "terminate" in ev or "termination" in ev:
            patterns += ["TerminationDate", "TerminatedDate", "EndDate", "EmploymentEndDate"]
        if "close" in ev or "closure" in ev:
            patterns += ["ClosedDate", "CloseDate", "ClosureDate", "DateClosed"]
        if "expire" in ev or "expiration" in ev:
            patterns += ["ExpirationDate", "ExpiryDate", "ExpiresOn"]
        if "complete" in ev or "completion" in ev:
            patterns += ["CompletedDate", "CompletionDate", "DateCompleted"]

        # de-dup keep order
        return list(dict.fromkeys(patterns))

    # -----------------------------
    # Excel profiling (Schema sheet)
    # -----------------------------
    def _profile_excel_schema(self, file_path: str, file_name: str) -> List[TableProfile]:
        current_app.logger.info(f"RCCEngine: profiling Excel schema for {file_name}")

        xls = pd.read_excel(file_path, sheet_name=None)

        schema_sheet_name = None
        for name in xls.keys():
            if str(name).strip().lower() == "schema":
                schema_sheet_name = name
                break
        if schema_sheet_name is None:
            current_app.logger.info(f"RCCEngine: no 'Schema' sheet found in {file_name}; skipping.")
            return []

        schema_df = xls[schema_sheet_name]

        # normalize col map
        cols = {str(c).strip().upper(): c for c in schema_df.columns}
        required = ["TABLE_CATALOG", "TABLE_SCHEMA", "TABLE_NAME", "COLUMN_NAME", "DATA_TYPE"]
        missing = [c for c in required if c not in cols]
        if missing:
            current_app.logger.info(f"RCCEngine: Schema sheet in {file_name} missing columns: {missing}")
            return []

        # Build a map of sheet name (lowered) -> DataFrame for data lookup
        sheet_map_lower = {str(k).strip().lower(): k for k in xls.keys()}

        profiles: List[TableProfile] = []
        for table_name, group in schema_df.groupby(cols["TABLE_NAME"]):
            table_catalog = group[cols["TABLE_CATALOG"]].iloc[0] if cols["TABLE_CATALOG"] in group else None
            table_schema = group[cols["TABLE_SCHEMA"]].iloc[0] if cols["TABLE_SCHEMA"] in group else None

            columns: List[TableColumnProfile] = []
            for _, r in group.iterrows():
                col_name = str(r[cols["COLUMN_NAME"]])
                data_type = None
                if pd.notna(r[cols["DATA_TYPE"]]):
                    data_type = str(r[cols["DATA_TYPE"]])

                is_nullable = None
                if "IS_NULLABLE" in cols and pd.notna(r[cols["IS_NULLABLE"]]):
                    raw = str(r[cols["IS_NULLABLE"]]).strip().upper()
                    is_nullable = raw == "YES"

                ordinal_position = None
                if "ORDINAL_POSITION" in cols and pd.notna(r[cols["ORDINAL_POSITION"]]):
                    try:
                        ordinal_position = int(r[cols["ORDINAL_POSITION"]])
                    except Exception:
                        ordinal_position = None

                column_default = None
                if "COLUMN_DEFAULT" in cols and pd.notna(r[cols["COLUMN_DEFAULT"]]):
                    column_default = str(r[cols["COLUMN_DEFAULT"]])

                char_max_length = None
                if "CHARACTER_MAXIMUM_LENGTH" in cols and pd.notna(r[cols["CHARACTER_MAXIMUM_LENGTH"]]):
                    try:
                        char_max_length = int(r[cols["CHARACTER_MAXIMUM_LENGTH"]])
                    except Exception:
                        char_max_length = None

                numeric_precision = None
                if "NUMERIC_PRECISION" in cols and pd.notna(r[cols["NUMERIC_PRECISION"]]):
                    try:
                        numeric_precision = int(r[cols["NUMERIC_PRECISION"]])
                    except Exception:
                        numeric_precision = None

                numeric_scale = None
                if "NUMERIC_SCALE" in cols and pd.notna(r[cols["NUMERIC_SCALE"]]):
                    try:
                        numeric_scale = int(r[cols["NUMERIC_SCALE"]])
                    except Exception:
                        numeric_scale = None

                date_time_precision = None
                if "DATETIME_PRECISION" in cols and pd.notna(r[cols["DATETIME_PRECISION"]]):
                    try:
                        date_time_precision = int(r[cols["DATETIME_PRECISION"]])
                    except Exception:
                        date_time_precision = None

                columns.append(
                    TableColumnProfile(
                        name=col_name,
                        data_type=data_type,
                        is_nullable=is_nullable,
                        ordinal_position=ordinal_position,
                        column_default=column_default,
                        char_max_length=char_max_length,
                        numeric_precision=numeric_precision,
                        numeric_scale=numeric_scale,
                        date_time_precision=date_time_precision,
                        sample_values=[],
                    )
                )

            # ── Load sample data from matching data sheet ──
            # Try to find a sheet whose name matches the table name
            tname_lower = str(table_name).strip().lower()
            sample_rows: List[Dict[str, str]] = []
            row_count = 0

            matched_sheet = sheet_map_lower.get(tname_lower)
            if matched_sheet:
                try:
                    data_df = xls[matched_sheet]
                    row_count = len(data_df)
                    # Take up to 5 sample rows
                    sample_df = data_df.head(5)
                    sample_rows = [
                        {str(k): str(v) for k, v in row_items.items()}
                        for row_items in sample_df.to_dict(orient="records")
                    ]
                    # Also populate sample_values per column
                    for col_profile in columns:
                        if col_profile.name in data_df.columns:
                            unique_vals = data_df[col_profile.name].dropna().head(5).astype(str).tolist()
                            col_profile.sample_values = unique_vals
                    current_app.logger.info(
                        f"RCCEngine: loaded {row_count} rows ({len(sample_rows)} samples) from sheet '{matched_sheet}' for table '{table_name}'"
                    )
                except Exception as e:
                    current_app.logger.warning(f"RCCEngine: error reading data sheet '{matched_sheet}': {e}")

            profiles.append(
                TableProfile(
                    file_name=file_name,
                    table_catalog=str(table_catalog) if table_catalog is not None else None,
                    table_schema=str(table_schema) if table_schema is not None else None,
                    table_name=str(table_name),
                    source_sheet_name=str(schema_sheet_name),
                    columns=columns,
                    row_count=row_count,
                    sample_rows=sample_rows,
                )
            )

        return profiles

    # -----------------------------
    # Classification (LLM-powered via GPT-4o-mini)
    # -----------------------------
    def _classify_profiles(
        self, profiles: List[TableProfile], defs: List[RCCDefinition]
    ) -> Dict[str, List[RCCClassificationResult]]:
        from app.llm_model import LLMModel
        from pydantic import BaseModel as PydanticBase, Field

        defs_by_code = {d.code.upper(): d for d in defs}

        # ── Pydantic model for LLM output (per table) ──
        class RCCMatchLLMOutput(PydanticBase):
            best_rcc_code: str = Field(description="The RCC code that best matches this table (e.g. ADM150)")
            confidence: float = Field(description="Confidence 0.0-1.0 that this is the correct match")
            reason: str = Field(description="2-4 sentence explanation citing specific evidence from table name, columns, data, and RCC definition")
            trigger_column: str = Field(description="Exact column name from the table to use as retention date trigger, or 'None' if not found")
            trigger_reason: str = Field(description="Why this column was chosen as the retention trigger")
            alternatives: List[Dict[str, Any]] = Field(default_factory=list, description="Up to 2 alternative RCC codes with code, confidence, and reason")

        # Initialize LLM (gpt-4o-mini for speed + cost)
        llm = LLMModel(model_source="gpt4o_mini")

        results_by_excel: Dict[str, List[RCCClassificationResult]] = {}

        # ── Build RCC reference block (same for all tables) ──
        rcc_ref_parts = []
        for d in defs:
            examples_str = ", ".join(d.record_type_examples[:10]) if d.record_type_examples else "None"
            trigger_str = ", ".join(d.trigger_column_name_patterns[:8]) if d.trigger_column_name_patterns else "None"
            rcc_ref_parts.append(
                f"RCC CODE: {d.code}\n"
                f"  Name: {d.name or 'N/A'}\n"
                f"  Description: {(d.description or 'N/A')[:300]}\n"
                f"  Record Type Examples: {examples_str}\n"
                f"  Retention Event: {d.retention_event or 'N/A'}\n"
                f"  Retention Period: {d.retention_period_years or '?'} year(s) {d.retention_period_months or 0} month(s)\n"
                f"  Trigger Column Patterns: {trigger_str}\n"
                f"  Jurisdiction: {d.jurisdiction or 'N/A'}"
            )
        rcc_reference = "\n\n".join(rcc_ref_parts)

        for profile in profiles:
            try:
                # ── Build table info block ──
                col_details = []
                for c in profile.columns[:30]:  # Limit to 30 columns
                    sample_str = ""
                    if c.sample_values:
                        sample_str = f" | samples: {', '.join(c.sample_values[:3])}"
                    col_details.append(f"  - {c.name} ({c.data_type or 'unknown'}){sample_str}")
                col_block = "\n".join(col_details)

                sample_block = ""
                if profile.sample_rows:
                    rows_str = []
                    for i, row in enumerate(profile.sample_rows[:3]):
                        truncated = {k: str(v)[:60] for k, v in list(row.items())[:10]}
                        rows_str.append(f"  Row {i+1}: {truncated}")
                    sample_block = f"\nSample Data:\n" + "\n".join(rows_str)

                prompt = f"""You are a records management expert. Match this database table to the BEST Record Class Code (RCC).

STRICT RULES - READ CAREFULLY:
- You MUST pick the RCC whose Description and Record Type Examples BEST match the table's purpose
- Evaluate: (1) table name meaning, (2) column names and types, (3) sample data patterns
- The table name + column structure reveals what KIND of data this table stores
- Match that PURPOSE to the RCC Description and Record Type Examples
- confidence must reflect ACTUAL match quality:
  * 0.85-1.0 = table clearly fits the RCC description and examples
  * 0.65-0.84 = strong alignment with minor uncertainty
  * 0.45-0.64 = moderate match, some signals align
  * below 0.45 = weak match
- For trigger_column: find the column that matches the RCC's retention event timing
  * If retention event is about creation date → look for CreatedDate, CreateDate, InsertDate, DateCreated, etc.
  * If about termination → look for TerminationDate, EndDate, etc.
  * Match ACTUAL column names from the table, not invented ones
- DO NOT hallucinate. If no RCC fits well, give low confidence. Only use information provided below.
- alternatives: provide up to 2 other RCC codes that could apply, with confidence and reason

=== RCC DEFINITIONS ===
{rcc_reference}

=== TABLE TO CLASSIFY ===
Table Name: {profile.table_name}
Schema: {profile.table_schema or 'N/A'}
Row Count: {profile.row_count}

Columns:
{col_block}
{sample_block}

Based on the table name, column structure, and sample data, which RCC code BEST describes what this table contains?"""

                current_app.logger.info(f"[RCC-LLM] Classifying table '{profile.table_name}' via GPT-4o-mini...")

                result: RCCMatchLLMOutput = llm.infer_model(prompt, RCCMatchLLMOutput)

                current_app.logger.info(
                    f"[RCC-LLM] TABLE='{profile.table_name}' → {result.best_rcc_code} "
                    f"(confidence={result.confidence:.0%}) | trigger={result.trigger_column} | "
                    f"reason={result.reason[:120]}"
                )

                # ── Build candidates from LLM output ──
                best_code = result.best_rcc_code.upper()
                if best_code not in defs_by_code:
                    current_app.logger.warning(f"[RCC-LLM] LLM returned unknown code '{best_code}', skipping")
                    continue

                selected_def = defs_by_code[best_code]

                # Validate trigger column exists in table
                valid_cols = {c.name.lower(): c.name for c in profile.columns}
                trigger_col = None
                if result.trigger_column and result.trigger_column.lower() != "none":
                    if result.trigger_column.lower() in valid_cols:
                        trigger_col = valid_cols[result.trigger_column.lower()]
                    else:
                        # Fallback: deterministic trigger detection
                        trigger_col = self._find_trigger_column(profile, selected_def)

                best_candidate = RCCCandidate(
                    code=best_code,
                    score=round(result.confidence, 4),
                    confidence=round(result.confidence, 4),
                    selected=True,
                    reason=result.reason,
                    trigger_column=trigger_col,
                    trigger_reason=result.trigger_reason,
                    retention_event=selected_def.retention_event,
                    retention_period_years=selected_def.retention_period_years,
                    retention_period_months=selected_def.retention_period_months,
                )

                candidates = [best_candidate]

                # Add alternatives from LLM
                for alt in (result.alternatives or [])[:2]:
                    alt_code = str(alt.get("code", "")).upper()
                    if alt_code and alt_code in defs_by_code and alt_code != best_code:
                        candidates.append(RCCCandidate(
                            code=alt_code,
                            score=round(float(alt.get("confidence", 0.0)), 4),
                            confidence=round(float(alt.get("confidence", 0.0)), 4),
                            selected=False,
                            not_selected_reason=alt.get("reason", ""),
                        ))

                classification = RCCClassificationResult(
                    file_name=profile.file_name,
                    table_name=profile.table_name,
                    best_code=best_code,
                    candidates=candidates,
                )

                results_by_excel.setdefault(profile.file_name.lower(), []).append(classification)

            except Exception as e:
                current_app.logger.error(f"[RCC-LLM] Error classifying table '{profile.table_name}': {e}", exc_info=True)
                continue

        return results_by_excel

    def _build_selected_reason(self, profile: TableProfile, d: RCCDefinition, best: RCCCandidate) -> str:
        bullets: List[str] = []

        # Emphasize Description-based matching
        if d.description:
            # Extract key matching terms
            table_tokens = set(self._tokenize(profile.table_name or ""))
            desc_tokens = set(self._tokenize(d.description or ""))
            matching_terms = table_tokens & desc_tokens
            
            if matching_terms:
                terms_str = ", ".join(sorted(matching_terms)[:5])  # Show up to 5 terms
                bullets.append(
                    f"Table '{profile.table_name}' strongly aligns with RCC {d.code} description. "
                    f"Matching key terms: {terms_str}."
                )
            else:
                bullets.append(
                    f"Table '{profile.table_name}' semantically aligns with RCC {d.code} description based on keyword overlap."
                )
        else:
            bullets.append(
                f"Table '{profile.table_name}' aligns with RCC {d.code} based on derived metadata."
            )

        # Column-level alignment
        col_tokens = set(self._tokenize(" ".join([c.name for c in profile.columns])))
        key_tokens = set(d.keywords or [])
        col_matches = col_tokens & key_tokens
        if col_matches:
            col_terms_str = ", ".join(sorted(col_matches)[:4])
            bullets.append(f"Column structure supports this classification (matching: {col_terms_str}).")

        # Retention/trigger alignment
        if self._is_date_driven(d.retention_event) and self._has_datetime_col(profile):
            bullets.append(f"Table contains date/time columns suitable for retention event '{d.retention_event}'.")

        return " ".join(bullets)

    def _build_not_selected_reason(
        self, profile: TableProfile, d: RCCDefinition, best: RCCCandidate, other: RCCCandidate
    ) -> str:
        # Simple deterministic comparative explanation
        msg = (
            f"Lower match score than selected {best.code} "
            f"({other.score:.2f} vs {best.score:.2f}) based on weaker overlap with table/column naming."
        )
        if self._is_date_driven(d.retention_event) and not self._has_datetime_col(profile):
            msg += " Also expected date-driven trigger signals are weaker."
        return msg

    def _build_trigger_reason(
        self, profile: TableProfile, d: RCCDefinition, trigger_col: Optional[str]
    ) -> Optional[str]:
        if not trigger_col:
            return "No suitable trigger column detected deterministically."
        # explain whether pattern-based or fallback
        patterns = [p.lower() for p in (d.trigger_column_name_patterns or [])]
        if any(p in trigger_col.lower() for p in patterns):
            return f"Matched RCC trigger pattern for event '{d.retention_event}'."
        return f"Fallback to date/time column because RCC event is '{d.retention_event}'."

    def _find_trigger_column(self, profile: TableProfile, definition: Optional[RCCDefinition]) -> Optional[str]:
        names = [c.name for c in profile.columns]
        lower_map = {n.lower(): n for n in names}

        # 1) RCC-specific trigger patterns
        if definition and definition.trigger_column_name_patterns:
            for pat in definition.trigger_column_name_patterns:
                pat_lower = str(pat).lower()
                for col_lower, original in lower_map.items():
                    if pat_lower and pat_lower in col_lower:
                        return original

        # 2) Fallback: any datetime-ish column (if present)
        for col in profile.columns:
            if col.data_type and "date" in col.data_type.lower():
                return col.name

        return None

    # -----------------------------
    # Utilities
    # -----------------------------
    def _tokenize(self, s: str) -> List[str]:
        if not s:
            return []
        import re
        s = s.replace("_", " ")
        parts = re.split(r"[^A-Za-z0-9]+", s)
        tokens: List[str] = []
        for p in parts:
            p = p.strip().lower()
            if not p:
                continue
            # split camelCase-ish
            p2 = re.sub(r"([a-z])([A-Z])", r"\1 \2", p)
            for t in p2.split():
                t = t.strip().lower()
                if t:
                    tokens.append(t)
        return tokens

    def _recall(self, query_tokens: set, reference_tokens: set) -> float:
        """
        Asymmetric recall: what fraction of query_tokens appear in reference_tokens?
        This avoids the Jaccard problem where a large reference set dilutes scores.
        """
        if not query_tokens or not reference_tokens:
            return 0.0
        hits = len(query_tokens & reference_tokens)
        return hits / len(query_tokens)

    def _jaccard(self, a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    def _is_date_driven(self, retention_event: Optional[str]) -> bool:
        if not retention_event:
            return False
        ev = retention_event.lower()
        return any(k in ev for k in ["create", "created", "close", "closed", "terminate", "termination", "expire", "expiration", "complete", "completion"])

    def _has_datetime_col(self, profile: TableProfile) -> bool:
        for col in profile.columns:
            if col.data_type and "date" in col.data_type.lower():
                return True
        return False

    # -----------------------------
    # Cache writers (debugging)
    # -----------------------------
    def _save_definitions_to_cache(self, defs: List[RCCDefinition]) -> None:
        try:
            cache_dir = Path(__file__).parent.parent / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            out_path = cache_dir / "rcc_definitions.json"

            existing: List[Dict[str, Any]] = []
            if out_path.exists():
                try:
                    existing = json.loads(out_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    existing = []

            new_entries = [d.model_dump() for d in defs]
            combined = existing + new_entries
            out_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
            current_app.logger.info(
                f"RCCEngine: wrote {len(new_entries)} RCC definition(s) to {out_path} (total {len(combined)})"
            )
        except Exception as e:
            current_app.logger.error(f"RCCEngine: error saving definitions: {e}")

    def _save_profiles_to_cache(self, profiles: List[TableProfile]) -> None:
        try:
            cache_dir = Path(__file__).parent.parent / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            out_path = cache_dir / "rcc_table_profiles.json"

            existing: List[Dict[str, Any]] = []
            if out_path.exists():
                try:
                    existing = json.loads(out_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    existing = []

            new_entries = [p.model_dump() for p in profiles]
            combined = existing + new_entries
            out_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
            current_app.logger.info(
                f"RCCEngine: wrote {len(new_entries)} table profile(s) to {out_path} (total {len(combined)})"
            )
        except Exception as e:
            current_app.logger.error(f"RCCEngine: error saving profiles: {e}")
