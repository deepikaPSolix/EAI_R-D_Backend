# app/rcc_explainer.py
from __future__ import annotations

import os
from typing import Optional, Dict, Any, List

from flask import current_app
from pydantic import BaseModel, Field

from app.llm_model import LLMModel


class RCCLLMExplanation(BaseModel):
    selected_reason_bullets: List[str] = Field(default_factory=list)
    not_selected: List[Dict[str, Any]] = Field(default_factory=list)
    trigger_explanation: str = ""
    confidence_explanation: str = ""


class RCCExplainerLLM:
    """
    Optional LLM explainer.
    - enabled only if RCC_USE_LLM_EXPLANATIONS=true
    - LLM ONLY rewrites/explains deterministically-selected decisions (grounded in evidence)
    - If LLM output invalid or tries to alter code/trigger -> ignore.
    """

    def __init__(self) -> None:
        self.enabled = str(os.getenv("RCC_USE_LLM_EXPLANATIONS", "false")).lower() in ("1", "true", "yes")
        self.max_alternatives = int(os.getenv("RCC_LLM_ALT_COUNT", "2"))
        self.model_source = os.getenv("RCC_LLM_SOURCE", "openai")  # openai|together|ollama|qwen|llama4|...

        if not self.enabled:
            self._llm = None
            return

        try:
            # json_mode not strictly required because you already parse/clean,
            # but it's nice to keep outputs JSONish where supported.
            self._llm = LLMModel(model_source=self.model_source, json_mode=True)
        except Exception as e:
            current_app.logger.warning(f"RCCExplainerLLM: LLM init failed, disabling. err={e}")
            self.enabled = False
            self._llm = None

    def enrich_classification(self, profile, definition, classification, alternative_defs=None):
        if not self.enabled or self._llm is None:
            return classification

        # Identify selected + alternatives (limited)
        candidates = classification.candidates or []
        selected = None
        alternatives = []
        for c in candidates:
            if getattr(c, "selected", False):
                selected = c
            else:
                alternatives.append(c)
        alternatives = alternatives[: self.max_alternatives]

        if not selected:
            return classification

        evidence = self._build_evidence_packet(profile, definition, selected, alternatives, alternative_defs or [])
        prompt = self._build_prompt(evidence)

        try:
            parsed: RCCLLMExplanation = self._llm.infer_model(prompt, RCCLLMExplanation)
        except Exception as e:
            current_app.logger.warning(f"RCCExplainerLLM: inference failed; fallback deterministic. err={e}")
            return classification

        # Hard safety: never allow LLM to change selected code/trigger
        # It may only provide text explanations.
        try:
            if isinstance(parsed.selected_reason_bullets, list) and parsed.selected_reason_bullets:
                selected.reason = " ".join([f"- {b}" for b in parsed.selected_reason_bullets if isinstance(b, str)])

            if isinstance(parsed.trigger_explanation, str) and parsed.trigger_explanation.strip():
                selected.trigger_reason = parsed.trigger_explanation.strip()

            # alternatives: parsed.not_selected = [{"code":"...", "bullets":[...]}]
            if isinstance(parsed.not_selected, list) and parsed.not_selected:
                alt_map = {}
                for x in parsed.not_selected:
                    if isinstance(x, dict) and isinstance(x.get("code"), str):
                        alt_map[x["code"].upper()] = x

                for alt in alternatives:
                    block = alt_map.get(alt.code.upper())
                    if not block:
                        continue
                    bullets = block.get("bullets")
                    if isinstance(bullets, list) and bullets:
                        alt.not_selected_reason = " ".join([f"- {b}" for b in bullets if isinstance(b, str)])

            # Optional: attach the confidence explanation into the selected.reason tail
            if isinstance(parsed.confidence_explanation, str) and parsed.confidence_explanation.strip():
                selected.reason = (selected.reason or "").strip() + f"  (Confidence: {parsed.confidence_explanation.strip()})"

        except Exception as e:
            current_app.logger.warning(f"RCCExplainerLLM: failed to apply explanation fields: {e}")

        return classification

    def _build_evidence_packet(self, profile, definition, selected, alternatives, alternative_defs) -> Dict[str, Any]:
        # Build comprehensive evidence including descriptions, events, and detailed table info
        
        # Get column details with data types
        column_details = []
        for c in profile.columns:
            col_info = {
                "name": c.name,
                "data_type": c.data_type,
            }
            if c.is_nullable is not None:
                col_info["is_nullable"] = c.is_nullable
            column_details.append(col_info)
        
        # Build selected definition details
        selected_def_details = {
            "code": definition.code,
            "name": definition.name,
            "description": definition.description,  # FULL description for LLM analysis
            "retention_event": definition.retention_event,
            "retention_period_years": definition.retention_period_years,
            "retention_period_months": definition.retention_period_months,
            "record_type_examples": (definition.record_type_examples or [])[:20],
            "trigger_column_name_patterns": (definition.trigger_column_name_patterns or [])[:15],
            "keywords": (definition.keywords or [])[:40],
            "jurisdiction": definition.jurisdiction,
        }
        
        # Build alternative definitions with full details for comparison
        alt_defs_map = {d.code.upper(): d for d in alternative_defs} if alternative_defs else {}
        alternatives_details = []
        for a in alternatives:
            alt_def = alt_defs_map.get(a.code.upper())
            alt_info = {
                "code": a.code,
                "score": a.score,
                "confidence": getattr(a, "confidence", None),
                "not_selected_reason": a.not_selected_reason,
            }
            
            # Add full definition details if available
            if alt_def:
                alt_info["definition"] = {
                    "name": alt_def.name,
                    "description": alt_def.description,  # KEY: Full description for comparison
                    "retention_event": alt_def.retention_event,
                    "retention_period_years": alt_def.retention_period_years,
                    "retention_period_months": alt_def.retention_period_months,
                    "record_type_examples": (alt_def.record_type_examples or [])[:15],
                    "keywords": (alt_def.keywords or [])[:30],
                }
            
            alternatives_details.append(alt_info)
        
        return {
            "selected_code": selected.code,
            "selected_score": selected.score,
            "selected_confidence": getattr(selected, "confidence", None),
            "table_details": {
                "table_name": profile.table_name,
                "table_schema": profile.table_schema,
                "table_catalog": profile.table_catalog,
                "columns": column_details,
                "column_count": len(column_details),
                "has_datetime_columns": any("date" in (c.data_type or "").lower() for c in profile.columns),
            },
            "trigger_analysis": {
                "trigger_column_selected": selected.trigger_column,
                "trigger_rule_used": "Definition trigger patterns first, then datetime fallback",
                "available_datetime_columns": [
                    c.name for c in profile.columns 
                    if c.data_type and "date" in c.data_type.lower()
                ],
            },
            "selected_definition": selected_def_details,
            "alternatives": alternatives_details,
            "scoring_methodology": {
                "description": "Deterministic scoring based on: 55% table name match, 30% column match, 15% multi-token bonus + retention alignment",
                "description_based": "Primary matching uses Description field from RCC definition, falling back to record_type_examples if description unavailable",
            },
            "constraints": [
                "Do not change selected_code.",
                "Do not change trigger_column_selected.",
                "Only rewrite explanations using the evidence provided.",
                "Base explanations on Description field alignment, retention event compatibility, and column structure.",
            ],
        }

    def _build_prompt(self, evidence: Dict[str, Any]) -> str:
        # Enhanced prompt with focus on Description, Event, and Table details
        return f"""
You are an expert Records Management specialist explaining Record Class Code (RCC) classification decisions.

STRICT RULES:
- Use ONLY the evidence in INPUT JSON below
- Base explanations on THREE KEY FACTORS:
  1. RCC Description field alignment with table/column names
  2. Retention Event compatibility with table structure (especially datetime columns)
  3. Record Type Examples and Keywords matching
- Do NOT invent policy text, retention periods, or trigger patterns
- Do NOT change selected_code or trigger_column_selected
- Be specific: cite actual table names, column names, description phrases, and retention events
- If evidence is insufficient: say "Insufficient evidence in available metadata"

TASK - Provide detailed, evidence-based explanations:

1) **SELECTED CODE EXPLANATION** (3-5 bullet points):
   - Why does the RCC Description semantically match this table?
   - How do the table/column names align with the Description and record_type_examples?
   - Why is the Retention Event compatible with the table structure?
   - What specific keywords or domain terms drove the match?
   - Why is the confidence level what it is?

2) **ALTERNATIVES NOT SELECTED** (2-3 bullets per alternative):
   - Why does each alternative RCC's Description NOT match as well?
   - What specific misalignments exist in retention event or table structure?
   - What keywords or semantic signals are weaker compared to selected code?
   - Be comparative: explain why selected is BETTER than this alternative

3) **TRIGGER COLUMN EXPLANATION** (1-2 sentences):
   - Why was this specific column chosen as trigger?
   - How does it relate to the Retention Event?
   - Was it matched by pattern or fallback logic?

4) **CONFIDENCE JUSTIFICATION** (1-2 sentences):
   - Explain the confidence score (0-1 scale) in business terms
   - What factors increase/decrease confidence?
   - Is this a strong match or borderline case?

OUTPUT JSON FORMAT:
{{
  "selected_reason_bullets": [
    "RCC Description states '...' which aligns with table name '...' because...",
    "Retention Event '...' requires datetime column, and table has '...' column of type...",
    "Column names [...] match record_type_examples [...] indicating...",
    "Domain-specific terms [...] appear in both description and table structure...",
    "Confidence of X.XX indicates... because..."
  ],
  "not_selected": [
    {{
      "code": "ALT001",
      "bullets": [
        "RCC Description '...' does not align with table '...' because...",
        "Retention Event '...' expects [...] but table structure shows...",
        "Weaker keyword overlap: only [...] matched vs selected's [...]"
      ]
    }}
  ],
  "trigger_explanation": "Column '...' selected because Retention Event '...' requires tracking from [...]. This column matched trigger pattern '...' defined in RCC.",
  "confidence_explanation": "Confidence of X.XX (Y%) indicates [strong/moderate/weak] match because description alignment is [high/medium/low], retention event compatibility is [clear/partial/unclear], and keyword overlap is [extensive/moderate/limited]."
}}

INPUT JSON:
{evidence}

REMEMBER: Be specific, cite evidence, explain WHY not just WHAT. Focus on Description field as primary signal.
""".strip()
