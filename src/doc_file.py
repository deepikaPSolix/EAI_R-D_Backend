import json
import os
import subprocess
import numpy as np
from unstructured.partition.auto import partition
from unstructured.partition.text import partition_text
from unstructured.chunking.basic import chunk_elements
from pathlib import Path
import torch
import whisper
from moviepy import VideoFileClip
from video_ocr import VideoOcr

class DocFile:
    def __init__(self, file_path: str):
        self.file_name = os.path.basename(file_path)
        self.file_type = Path(self.file_name).suffix[1:]

        if self.file_type == "mp3":
           audio_text = self.extract_text_from_audio(file_path)
           elements = partition_text(text = audio_text)
           self.chunks = chunk_elements(elements, overlap=50, max_characters=2000)
        elif self.file_type == "mp4":
            video_text = ""
            audio_text = ""
            if self._has_audio(file_path):
                video_text = VideoOcr.extract_text_from_video(video_path=file_path, pixel_diff_threshold=0.5)
                # self._extract_audio(file_path)
                audio_text = self.extract_text_from_audio("cache/output/audio.mp3")
            else:
                video_text = VideoOcr.extract_text_from_video(video_path=file_path, pixel_diff_threshold=0.1)
            elements = partition_text(text = audio_text + video_text)
            self.chunks = chunk_elements(elements, overlap=50, max_characters=2000)
        else:
            self.chunks = self._extract_data_chunks(file_path)
        self.data = " | ".join([chunk.text for chunk in self.chunks])
        
    def extract_text_from_audio(self, file_path: str):
        print("GPU: ", torch.cuda.is_available())
        model = whisper.load_model("turbo")
        result = model.transcribe(file_path)
        return result['text']
        
    def _extract_data_chunks(self, file_path: str):
        elements = partition(filename=file_path)
        chunks = chunk_elements(elements, overlap=50, max_characters=2000)
        return chunks

    def _has_audio(self, file_path, silence_threshold=0.01):
        result = True
        try:
            clip = VideoFileClip(file_path)
            if clip.audio is not None:
                # Extract the audio and check if it's silent
                audio_array = clip.audio.to_soundarray(fps=44100)
                max_amplitude = np.max(np.abs(audio_array))
                
                if max_amplitude < silence_threshold:
                    result = False
                else:
                    clip.audio.write_audiofile("cache/output/audio.mp3")
            else:
                result = False
                    
        except FileNotFoundError as e:
            raise RuntimeError("File not found: " + str(e))
        except Exception as e:
            raise RuntimeError(f"An error occurred: {e}")
        finally:
            clip.close()

        return result
    
    def to_dict(self):
        return {
            "file_name": self.file_name,
            "data": self.data
        }
        
