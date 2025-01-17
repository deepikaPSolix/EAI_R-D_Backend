from abc import ABC, abstractmethod
import difflib
import re

from moviepy import AudioFileClip, VideoFileClip
import numpy as np
from paddleocr import PaddleOCR
import torch
import whisper
from unstructured.partition.text import partition_text
from unstructured.partition.auto import partition
from unstructured.chunking.basic import chunk_elements
from unstructured.documents.elements import Element

from app.video_ocr import VideoOcr

class FileProcessor(ABC):
    @abstractmethod
    def process_file(self, file_path:str) -> list[Element]:
        """Process the file and return chunks of data."""
        pass

    def chunks_from_text(self, text:str) -> list[Element] :
        """
        Splits the provided text into smaller chunks for processing.

        Args:
            text (str): The input text to be partitioned and chunked.

        Returns:
            list[Element]: A list of chunked elements derived from the input text.
        """

        elements = partition_text(text=text)
        return chunk_elements(elements, overlap=50, max_characters=2000)
    
    def chunks_from_file(self, file_path:str) -> list[Element] :
        """
        Processes a file to partition its contents into smaller, manageable chunks.

        Args:
            file_path (str): The path to the file to be processed.

        Returns:
            list[Element]: A list of chunked elements derived from the file content.
        """

        elements = partition(filename=file_path)
        return chunk_elements(elements, overlap=50, max_characters=2000)
    

class GenericFileProcessor(FileProcessor):
    def process_file(self, file_path) -> list[Element]:
        return super().chunks_from_file(file_path)


class AudioFileProcessor(FileProcessor):
    def process_file(self, file_path) -> list[Element]:
        audio_text = self.extract_text_from_audio(file_path)
        return super().chunks_from_text(audio_text)

    def extract_text_from_audio(self, file_path: str):
        print("GPU: ", torch.cuda.is_available())
        model = whisper.load_model("turbo")
        result = model.transcribe(file_path)
        return result['text']
    
class VideoFileProcessor(AudioFileProcessor):

    def __init__(self):
        self.__ocr = PaddleOCR(use_angle_cls=True, lang='en')

    def process_file(self, file_path):
        video_text = ""
        audio_text = ""
        try:
            clip = VideoFileClip(file_path)

            if self._has_audio(audioClip=clip.audio):
                clip.audio.write_audiofile("cache/output/audio.mp3")
                video_text = self.__extract_text_from_video(clip=clip, pixel_diff_threshold=0.5)
                audio_text = self.extract_text_from_audio("cache/output/audio.mp3")
            else:
                video_text = self.__extract_text_from_video(clip=clip, pixel_diff_threshold=0.1)
            return super().chunks_from_text(audio_text + video_text)
        finally:
            clip.close()

    def _has_audio(self, audioClip: AudioFileClip, silence_threshold=0.01):
        try:
            if not audioClip:
                return False
            
            audio_array = audioClip.to_soundarray(fps=44100)
            max_amplitude = np.max(np.abs(audio_array))

            if max_amplitude < silence_threshold:
                return False
            
            return True
                    
        except FileNotFoundError as e:
            raise RuntimeError("File not found: " + str(e))
        except Exception as e:
            raise RuntimeError(f"An error occurred: {e}")
        
    def __extract_text_from_video(self, clip: VideoFileClip, fps:int = 1, pixel_diff_threshold = 0.5):
        text_data = []
        prev_frame = None
        prev_text = ""

        frames = clip.iter_frames(fps=fps)
        for frame in frames:
            if prev_frame is not None:
                pixel_diff = self.__calculate_pixel_difference(prev_frame, frame)
                if pixel_diff < pixel_diff_threshold:
                    continue

            current_text = self.__process_frame(frame)

            if current_text and self.__is_similar(current_text, prev_text):
                text_data.append(current_text)
                prev_text = current_text
            prev_frame = frame


        return "".join(text_data)


    def __process_frame(self, frame):
        """
        Process a frame for OCR and extract text.

        Args:
            frame (numpy.ndarray): The current frame from the video.

        Returns:
            str: Extracted text from the frame.
        """
        try:
            result = self.__ocr.ocr(frame, det=True, cls=True)
            if result and len(result) > 0:
                text = [item[1][0] for item in result[0] if len(item) > 1 and item[1]]
                return ' '.join(text)
            else:
                return ""
        except Exception as e:
            print(f"Error during OCR processing: {e}")
            return ""
        
    def __calculate_pixel_difference(self, prev_frame, curr_frame):
        """
        Calculate the pixel difference between two frames.

        Args:
            prev_frame (numpy.ndarray): The previous frame.
            curr_frame (numpy.ndarray): The current frame.

        Returns:
            float: The percentage of pixel difference.
        """
        diff = np.abs(prev_frame -curr_frame)
        non_zero_diff = np.count_nonzero(diff)
        total_pixels = np.prod(diff.shape)  # Total number of pixels in the frame
        return non_zero_diff / total_pixels  # Return the fraction of different pixels

    def __is_similar(self, current_text, previous_text, threshold=0.7):
        """
        Compare two text strings and check if they are significantly similar using difflib.

        Args:
            current_text (str): The text from the current frame.
            previous_text (str): The text from the previous frame.
            threshold (float): The threshold above which the texts are considered similar.

        Returns:
            bool: True if the texts are similar, False otherwise.
        """
        if not previous_text:
            return False  # First frame, no comparison to be made

        # Normalize the text for better comparison
        current_text = self.__normalize_text(current_text)
        previous_text = self.__normalize_text(previous_text)
        similarity_ratio = difflib.SequenceMatcher(None, current_text, previous_text).ratio()
        return similarity_ratio > threshold

    def __normalize_text(self, text):
        """
        Normalize text by removing extra spaces, punctuation, and converting to lowercase.

        Args:
            text (str): The text to normalize.

        Returns:
            str: The normalized text.
        """
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

        