import os
from unstructured.partition.auto import partition
from unstructured.partition.text import partition_text
from unstructured.chunking.basic import chunk_elements
from pathlib import Path
import torch
import whisper
from ffmpeg import FFmpeg
# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

class DocFile:
    def __init__(self, file_path: str):
        self.file_name = os.path.basename(file_path)
        self.file_type = Path(self.file_name).suffix[1:]
        if self.file_type == "mp3":
           self.chunks = self.extract_text_from_audio(file_path)
        elif self.file_type == "mp4":
            self._extract_audio(file_path)
            self.chunks = self.extract_text_from_audio("cache/output/audio.mp3")
        else:
            self.chunks = self._extract_data_chunks(file_path)
        self.data = " | ".join([chunk.text for chunk in self.chunks])
        
    def extract_text_from_audio(self, file_path: str):
        print("GPU: ", torch.cuda.is_available())
        model = whisper.load_model("turbo")
        result = model.transcribe(file_path)
        text = result['text']
        elements = partition_text(text = text)
        chunks = chunk_elements(elements, overlap=50, max_characters=2000)
        return chunks
        
    def _extract_data_chunks(self, file_path: str):
        elements = partition(filename=file_path)
        chunks = chunk_elements(elements, overlap=50, max_characters=2000)
        return chunks
    
    def _extract_audio(self, file_path:str):
        ffmpeg = (
            FFmpeg()
            .option("y")  # Overwrite output file if it exists
            .input(file_path)
            .output(
                "cache/output/audio.mp3",
                **{"q:a": 0},  # Set audio quality
                **{"map": "a"}  # Extract only the audio stream
            )
        )

        ffmpeg.execute()
    
    def to_dict(self):
        return {
            "file_name": self.file_name,
            "data": self.data
        }
        
