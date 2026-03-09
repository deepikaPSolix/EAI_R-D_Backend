"""
RCC Matcher - LLM-Powered Table to RCC Code Matching
Compares Excel tables against RCC definitions using intelligent reasoning
"""

from typing import Dict, List, Optional, Tuple
from flask import current_app
from app.models.label_governance_model import AttributesModel
from app.llm_model import LLMModel
from pydantic import BaseModel, Field
import re


class RCCMatchResult(BaseModel):
    """Result of RCC matching for a single table"""
    table_name: str
    rcc_code: str
    rcc_name: str
    retention_period: str
    retention_trigger_column: str
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence between 0 and 1")
    match_reasons: List[str] = Field(default_factory=list, description="Reasons for this RCC assignment")
    is_match: bool = Field(True, description="Whether a match was found")


class RCCMatcher:
    """Match Excel tables to RCC definitions using LLM-based analysis"""
    
    def __init__(self):
        self.llm = LLMModel()
    
    def match_table_to_rccs(
        self, 
        table_info: Dict, 
        rcc_definitions: List[AttributesModel]
    ) -> RCCMatchResult:
        """
        Match a single table to the best RCC definition
        
        Args:
            table_info: Table dictionary from ExcelSchemaParser
            rcc_definitions: List of all available RCC definitions
            
        Returns:
            RCCMatchResult with best match and reasoning
        """
        if not rcc_definitions:
            current_app.logger.warning("No RCC definitions available for matching")
            return self._no_match_result(table_info['table_name'])
        
        # Build matching prompt
        prompt = self._build_matching_prompt(table_info, rcc_definitions)
        
        try:
            # Get LLM's analysis
            result = self.llm.infer_model(prompt, RCCMatchResult)
            
            if result and result.is_match:
                current_app.logger.info(
                    f"✅ Matched table '{table_info['table_name']}' to RCC {result.rcc_code} "
                    f"(confidence: {result.confidence_score:.2f})"
                )
                
                # If no trigger column identified, try pattern matching
                if not result.retention_trigger_column or result.retention_trigger_column == "Unknown":
                    result.retention_trigger_column = self._find_trigger_column(
                        table_info, 
                        next((rcc for rcc in rcc_definitions if rcc.record_class_code == result.rcc_code), None)
                    )
                
                return result
            else:
                return self._no_match_result(table_info['table_name'])
                
        except Exception as e:
            current_app.logger.error(f"Error matching table {table_info['table_name']}: {str(e)}")
            return self._no_match_result(table_info['table_name'])
    
    def _build_matching_prompt(self, table_info: Dict, rcc_definitions: List[AttributesModel]) -> str:
        """
        Build comprehensive prompt for LLM to match table to RCC
        """
        # Table summary
        table_summary = f"""
**TABLE TO ANALYZE:**
Table Name: {table_info['table_name']}
Columns ({len(table_info['column_names'])}): {', '.join(table_info['column_names'])}

Column Details:
{self._format_columns(table_info['columns'])}

Sample Data (first 3 rows):
{self._format_sample_data(table_info['sample_data'])}
"""
        
        # RCC definitions summary
        rcc_summary = "\n\n".join([
            self._format_rcc_definition(rcc) for rcc in rcc_definitions
        ])
        
        prompt = f"""
**TASK: Match Database Table to Record Class Code (RCC)**

You are an expert in data governance and records management. Your task is to analyze a database table and determine which Record Class Code (RCC) best applies to it.

{table_summary}

---

**AVAILABLE RCC DEFINITIONS:**

{rcc_summary}

---

**YOUR TASK:**

1. **Compare the table** against EACH RCC definition above
2. **Consider multiple factors:**
   - Does the table name match any Record Type Examples?
   - Do the column names suggest the type of data covered by the RCC?
   - Does the sample data content align with the RCC description?
   - Does the retention event make sense for this table's purpose?

3. **Select the BEST MATCH:**
   - Choose the RCC code that most closely matches this table
   - If NO good match exists, set is_match=false
   - If multiple RCCs could apply, choose the most specific one

4. **Identify Retention Trigger Column:**
   - Based on the RCC's retention_event and trigger_column_patterns
   - Find the actual column name in the table that should trigger retention
   - Look for exact matches or close variations (case-insensitive, with/without underscores)
   - Example: If RCC needs "CreatedDate", table might have "Created_Date" or "CreateDate"

5. **Provide Confidence Score (0.0 to 1.0):**
   - 0.9-1.0: Very strong match (table name + columns + data all align)
   - 0.75-0.89: Good match (most factors align)
   - 0.6-0.74: Moderate match (some factors align)
   - Below 0.6: Weak match (consider setting is_match=false)

6. **Explain Your Reasoning:**
   - Provide 2-5 specific reasons for your choice
   - Reference exact matches (e.g., "Table name 'LogEmails' matches RCC example 'Email Logs'")
   - Be specific about what convinced you

**OUTPUT FORMAT:**
Return a JSON object with:
- table_name: (the table name)
- rcc_code: (e.g., "ADM150") or empty string if no match
- rcc_name: (e.g., "Internal Tracking/Monitoring") or empty string
- retention_period: (e.g., "1 Year") or empty string
- retention_trigger_column: (exact column name from table, or "Unknown" if not found)
- confidence_score: (float between 0.0 and 1.0)
- match_reasons: (array of specific reasons)
- is_match: (true if good match found, false otherwise)

**IMPORTANT:**
- Be strict in your matching - only assign RCC if there's genuine alignment
- If unsure, set is_match=false rather than forcing a weak match
- Provide detailed reasoning so users understand your decision
"""
        
        return prompt
    
    def _format_columns(self, columns: List[Dict]) -> str:
        """Format column list for prompt"""
        if not columns:
            return "  (No columns available)"
        
        formatted = []
        for col in columns[:15]:  # Limit to 15 columns
            formatted.append(f"  - {col['column_name']} ({col.get('data_type', 'unknown')})")
        
        if len(columns) > 15:
            formatted.append(f"  ... and {len(columns) - 15} more columns")
        
        return "\n".join(formatted)
    
    def _format_sample_data(self, sample_data: List[Dict]) -> str:
        """Format sample data for prompt"""
        if not sample_data:
            return "  (No sample data available)"
        
        formatted = []
        for i, row in enumerate(sample_data[:3]):
            # Truncate long values
            row_str = {k: str(v)[:50] + ('...' if len(str(v)) > 50 else '') for k, v in row.items()}
            formatted.append(f"  Row {i+1}: {row_str}")
        
        return "\n".join(formatted)
    
    def _format_rcc_definition(self, rcc: AttributesModel) -> str:
        """Format a single RCC definition for the prompt"""
        formatted = f"""
**RCC CODE: {rcc.record_class_code}**
Name: {rcc.record_class_name}
Description: {rcc.attributes.get('description', 'N/A') if rcc.attributes else 'N/A'}
Retention Period: {rcc.retention_time}
Retention Event: {rcc.retention_event}
Jurisdiction: {rcc.jurisdiction}

Record Type Examples:
{chr(10).join(['  - ' + example for example in rcc.record_type_examples]) if rcc.record_type_examples else '  (None)'}

Trigger Column Patterns to look for:
{', '.join(rcc.trigger_column_patterns[:8]) if rcc.trigger_column_patterns else '(None)'}
"""
        return formatted
    
    def _find_trigger_column(self, table_info: Dict, rcc: Optional[AttributesModel]) -> str:
        """
        Fallback pattern matching to find trigger column if LLM didn't identify it
        
        Args:
            table_info: Table information
            rcc: RCC definition (if matched)
            
        Returns:
            Column name or "Unknown"
        """
        if not rcc or not rcc.trigger_column_patterns:
            return "Unknown"
        
        column_names = table_info['column_names']
        
        # Normalize columns for comparison
        normalized_columns = {self._normalize_column_name(col): col for col in column_names}
        
        # Try each pattern
        for pattern in rcc.trigger_column_patterns:
            normalized_pattern = self._normalize_column_name(pattern)
            
            # Exact match after normalization
            if normalized_pattern in normalized_columns:
                return normalized_columns[normalized_pattern]
            
            # Partial match
            for norm_col, original_col in normalized_columns.items():
                if normalized_pattern in norm_col or norm_col in normalized_pattern:
                    return original_col
        
        # Look for common date columns
        date_keywords = ['date', 'time', 'timestamp', 'dt']
        for col in column_names:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in date_keywords):
                return col
        
        return "Unknown"
    
    def _normalize_column_name(self, name: str) -> str:
        """Normalize column name for comparison (lowercase, no separators)"""
        return re.sub(r'[_\s-]', '', name.lower())
    
    def _no_match_result(self, table_name: str) -> RCCMatchResult:
        """Create a no-match result"""
        return RCCMatchResult(
            table_name=table_name,
            rcc_code="",
            rcc_name="",
            retention_period="",
            retention_trigger_column="",
            confidence_score=0.0,
            match_reasons=["No suitable RCC definition found for this table"],
            is_match=False
        )
