import os
from unstructured.partition.auto import partition
from unstructured.partition.text import partition_text
from unstructured.chunking.basic import chunk_elements
from pathlib import Path
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

class DocFile:
    def __init__(self, file_path: str):
        self.file_name = os.path.basename(file_path)
        self.file_type = Path(self.file_name).suffix[1:]
        if self.file_type == "mp3":
           self.chunks = self.extract_text_from_audio(file_path)
        else:
            self.chunks = self._extract_data_chunks(file_path)
        self.data = " | ".join([chunk.text for chunk in self.chunks])
        
    def extract_text_from_audio(self, file_path: str):
        device = 0 if torch.cuda.is_available() else -1  # Use 0 for GPU, -1 for CPU
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", device = device, torch_dtype=torch_dtype)
        result = pipe(file_path, return_timestamps=True)
        text = result['text']
        elements = partition_text(text = text)
        chunks = chunk_elements(elements, overlap=50, max_characters=2000)
        return chunks


         
        
    def _extract_data_chunks(self, file_path: str):
        elements = partition(filename=file_path)
        chunks = chunk_elements(elements, overlap=50, max_characters=2000)
        return chunks
    
    def to_dict(self):
        return {
            "file_name": self.file_name,
            "data": self.data
        }
        
