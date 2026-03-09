"""
Excel Schema Parser
Extracts table structures, column definitions, and sample data from multi-sheet Excel files
"""

import pandas as pd
from typing import Dict, List, Optional
from flask import current_app


class ExcelSchemaParser:
    """Parse Excel files to extract database schema information"""
    
    def parse_excel_file(self, file_path: str) -> Dict:
        """
        Parse Excel file and extract all sheets with metadata
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            Dictionary with structure:
            {
                'file_name': str,
                'sheet_names': List[str],
                'has_schema_sheet': bool,
                'has_tables_sheet': bool,
                'tables': List[Dict]  # Each table with schema + sample data
            }
        """
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            current_app.logger.info(f"📊 Parsing Excel file with sheets: {sheet_names}")
            
            # Detect special sheets
            has_schema_sheet = any('schema' in s.lower() for s in sheet_names)
            has_tables_sheet = any('table' in s.lower() and s.lower() != 'tables' for s in sheet_names)
            
            # Parse based on structure
            if has_schema_sheet:
                # Structured format with Schema sheet
                tables = self._parse_structured_excel(excel_file, sheet_names)
            else:
                # Generic format - each sheet is a table
                tables = self._parse_generic_excel(excel_file, sheet_names)
            
            return {
                'file_name': file_path.split('/')[-1].split('\\')[-1],
                'sheet_names': sheet_names,
                'has_schema_sheet': has_schema_sheet,
                'has_tables_sheet': has_tables_sheet,
                'tables': tables,
                'total_tables': len(tables)
            }
            
        except Exception as e:
            current_app.logger.error(f"Error parsing Excel file {file_path}: {str(e)}")
            raise
    
    def _parse_structured_excel(self, excel_file: pd.ExcelFile, sheet_names: List[str]) -> List[Dict]:
        """
        Parse Excel with Schema sheet (your format)
        
        Expected structure:
        - "Schema" sheet: TABLE_NAME, COLUMN_NAME, DATA_TYPE, etc.
        - "Tables" sheet: List of table names
        - Individual data sheets for each table
        """
        tables = []
        
        # Find Schema sheet
        schema_sheet_name = next((s for s in sheet_names if 'schema' in s.lower()), None)
        if not schema_sheet_name:
            return self._parse_generic_excel(excel_file, sheet_names)
        
        # Read Schema sheet
        schema_df = pd.read_excel(excel_file, sheet_name=schema_sheet_name)
        
        # Group by table name
        if 'TABLE_NAME' in schema_df.columns:
            table_names = schema_df['TABLE_NAME'].unique()
        else:
            # Fallback to generic parsing
            return self._parse_generic_excel(excel_file, sheet_names)
        
        # Parse each table
        for table_name in table_names:
            # Get schema for this table
            table_schema_df = schema_df[schema_df['TABLE_NAME'] == table_name]
            
            # Extract columns
            columns = []
            if 'COLUMN_NAME' in table_schema_df.columns:
                for _, row in table_schema_df.iterrows():
                    col_info = {
                        'column_name': row.get('COLUMN_NAME', ''),
                        'data_type': row.get('DATA_TYPE', ''),
                        'is_nullable': row.get('IS_NULLABLE', 'YES'),
                        'character_maximum_length': row.get('CHARACTER_MAXIMUM_LENGTH', None)
                    }
                    columns.append(col_info)
            
            # Try to find data sheet for this table
            data_sheet_name = self._find_data_sheet(sheet_names, table_name)
            sample_data = []
            row_count = 0
            
            if data_sheet_name:
                try:
                    data_df = pd.read_excel(excel_file, sheet_name=data_sheet_name, nrows=25)
                    sample_data = data_df.to_dict(orient='records')
                    row_count = len(data_df)
                except Exception as e:
                    current_app.logger.warning(f"Could not read data sheet {data_sheet_name}: {str(e)}")
            
            tables.append({
                'table_name': table_name,
                'columns': columns,
                'column_names': [c['column_name'] for c in columns],
                'sample_data': sample_data,
                'row_count': row_count,
                'has_sample_data': len(sample_data) > 0
            })
        
        return tables
    
    def _parse_generic_excel(self, excel_file: pd.ExcelFile, sheet_names: List[str]) -> List[Dict]:
        """
        Parse Excel where each sheet is a table (generic format)
        """
        tables = []
        
        # Skip common metadata sheets
        skip_sheets = ['schema', 'tables', 'metadata', 'readme', 'index']
        data_sheets = [s for s in sheet_names if s.lower() not in skip_sheets]
        
        for sheet_name in data_sheets:
            try:
                # Read sheet (max 25 rows for sample)
                df = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=25)
                
                if df.empty:
                    continue
                
                # Extract column info
                columns = []
                for col in df.columns:
                    col_info = {
                        'column_name': str(col),
                        'data_type': str(df[col].dtype),
                        'is_nullable': 'YES' if df[col].isnull().any() else 'NO',
                        'character_maximum_length': None
                    }
                    columns.append(col_info)
                
                # Get sample data
                sample_data = df.to_dict(orient='records')
                
                tables.append({
                    'table_name': sheet_name,
                    'columns': columns,
                    'column_names': list(df.columns),
                    'sample_data': sample_data,
                    'row_count': len(df),
                    'has_sample_data': True
                })
                
            except Exception as e:
                current_app.logger.warning(f"Could not parse sheet {sheet_name}: {str(e)}")
                continue
        
        return tables
    
    def _find_data_sheet(self, sheet_names: List[str], table_name: str) -> Optional[str]:
        """
        Find the data sheet corresponding to a table name
        
        Looks for exact match or partial match
        """
        # Exact match (case-insensitive)
        for sheet in sheet_names:
            if sheet.lower() == table_name.lower():
                return sheet
        
        # Partial match
        for sheet in sheet_names:
            if table_name.lower() in sheet.lower() or sheet.lower() in table_name.lower():
                # Avoid matching Schema/Tables sheets
                if 'schema' not in sheet.lower() and 'tables' not in sheet.lower():
                    return sheet
        
        return None
    
    def get_table_summary(self, table_info: Dict) -> str:
        """
        Generate a text summary of table for LLM matching
        
        Args:
            table_info: Table dictionary from parse_excel_file
            
        Returns:
            Formatted summary string
        """
        summary = f"Table Name: {table_info['table_name']}\n"
        summary += f"Columns ({len(table_info['columns'])}): {', '.join(table_info['column_names'])}\n"
        
        # Add column details
        summary += "\nColumn Details:\n"
        for col in table_info['columns'][:10]:  # Limit to first 10 columns
            summary += f"  - {col['column_name']} ({col['data_type']})\n"
        
        # Add sample data summary
        if table_info['sample_data']:
            summary += f"\nSample Data ({table_info['row_count']} rows):\n"
            # Show first 3 rows
            for i, row in enumerate(table_info['sample_data'][:3]):
                summary += f"  Row {i+1}: {str(row)[:200]}...\n"
        
        return summary
