"""
RCC Definition Storage Manager
Provides in-memory cache with persistent backup to cache/ folder
"""

import json
import os
from typing import Dict, List, Optional
from flask import current_app
from app.models.label_governance_model import AttributesModel


class RCCDefinitionCache:
    """
    Singleton in-memory cache for RCC definitions with persistent JSON backup
    """
    _instance = None
    _rcc_definitions: Dict[str, AttributesModel] = {}
    _cache_file = "cache/rcc_definitions.json"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_from_disk()
        return cls._instance
    
    def add_rcc(self, rcc_code: str, rcc_definition: AttributesModel):
        """
        Add or update an RCC definition
        
        Args:
            rcc_code: RCC code (e.g., "ADM150")
            rcc_definition: AttributesModel with is_rcc_document=True
        """
        if not rcc_definition.is_rcc_document:
            current_app.logger.warning(f"Attempted to add non-RCC document as RCC: {rcc_code}")
            return False
        
        self._rcc_definitions[rcc_code] = rcc_definition
        current_app.logger.info(f"✅ Added RCC definition: {rcc_code} - {rcc_definition.record_class_name}")
        
        # Persist to disk
        self._save_to_disk()
        return True
    
    def get_rcc(self, rcc_code: str) -> Optional[AttributesModel]:
        """
        Retrieve an RCC definition by code
        
        Args:
            rcc_code: RCC code (e.g., "ADM150")
            
        Returns:
            AttributesModel if found, None otherwise
        """
        return self._rcc_definitions.get(rcc_code)
    
    def get_all_rccs(self) -> List[AttributesModel]:
        """
        Get all stored RCC definitions
        
        Returns:
            List of AttributesModel objects
        """
        return list(self._rcc_definitions.values())
    
    def get_all_rcc_codes(self) -> List[str]:
        """
        Get all RCC codes
        
        Returns:
            List of RCC code strings
        """
        return list(self._rcc_definitions.keys())
    
    def has_rcc(self, rcc_code: str) -> bool:
        """Check if RCC code exists in cache"""
        return rcc_code in self._rcc_definitions
    
    def remove_rcc(self, rcc_code: str) -> bool:
        """
        Remove an RCC definition
        
        Args:
            rcc_code: RCC code to remove
            
        Returns:
            True if removed, False if not found
        """
        if rcc_code in self._rcc_definitions:
            del self._rcc_definitions[rcc_code]
            self._save_to_disk()
            current_app.logger.info(f"🗑️ Removed RCC definition: {rcc_code}")
            return True
        return False
    
    def clear_all(self):
        """Clear all RCC definitions from cache and disk"""
        self._rcc_definitions.clear()
        self._save_to_disk()
        current_app.logger.info("🗑️ Cleared all RCC definitions")
    
    def _save_to_disk(self):
        """Persist cache to JSON file in cache/ folder"""
        try:
            # Ensure cache directory exists
            os.makedirs("cache", exist_ok=True)
            
            # Convert to serializable dict
            data = {
                code: rcc.model_dump() 
                for code, rcc in self._rcc_definitions.items()
            }
            
            with open(self._cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            current_app.logger.debug(f"💾 Saved RCC definitions to {self._cache_file}")
        except Exception as e:
            current_app.logger.error(f"Failed to save RCC definitions to disk: {str(e)}")
    
    def _load_from_disk(self):
        """Load cache from JSON file on startup"""
        try:
            if os.path.exists(self._cache_file):
                with open(self._cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Reconstruct AttributesModel objects
                for code, rcc_dict in data.items():
                    self._rcc_definitions[code] = AttributesModel(**rcc_dict)
                
                if hasattr(current_app, 'logger'):
                    current_app.logger.info(f"📂 Loaded {len(self._rcc_definitions)} RCC definitions from disk")
                else:
                    print(f"📂 Loaded {len(self._rcc_definitions)} RCC definitions from {self._cache_file}")
        except Exception as e:
            if hasattr(current_app, 'logger'):
                current_app.logger.warning(f"Could not load RCC definitions from disk: {str(e)}")
            else:
                print(f"Warning: Could not load RCC definitions: {str(e)}")
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            "total_rccs": len(self._rcc_definitions),
            "rcc_codes": list(self._rcc_definitions.keys()),
            "cache_file": self._cache_file,
            "cache_file_exists": os.path.exists(self._cache_file)
        }
