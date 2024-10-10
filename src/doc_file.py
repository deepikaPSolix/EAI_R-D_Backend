import os
from unstructured.partition.auto import partition
from unstructured.chunking.basic import chunk_elements

class DocFile:
    def __init__(self, file_path: str):
        self.file_name = os.path.basename(file_path)
        self.chunks = self._extract_data_chunks(file_path)
        self.data = " | ".join([chunk.text for chunk in self.chunks])
        
    def _extract_data_chunks(self, file_path: str):
        elements = partition(filename=file_path)
        chunks = chunk_elements(elements, overlap=50, max_characters=2000)
        return chunks
    
    def to_dict(self):
        return {
            "file_name": self.file_name,
            "data": self.data
        }
        
