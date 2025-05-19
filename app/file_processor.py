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
from pptx import Presentation
import io
from docx import Document
from lxml import etree
import openpyxl
from flask import current_app
import pandas as pd

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

class ExcelFileProcessor(FileProcessor):
    def process_file(self, file_path) -> list[Element]:
        # Get clean text from Excel and chunk it
        excel_text = self.__extract_text_from_excel(file_path)
        return super().chunks_from_text(excel_text)

    def __extract_text_from_excel(self, file_path: str) -> str:
        df_sheets = pd.read_excel(file_path, sheet_name=None)
        text_data = []

        for sheet_name, df in df_sheets.items():
            df = df.dropna(how='all')           # Drop fully empty rows
            df = df.dropna(axis=1, how='all')   # Drop fully empty columns
            df = df.fillna("")                  # Replace remaining NaNs with empty strings

            if not df.empty:
                text = df.to_string(index=False)
                text_data.append(f"[Sheet: {sheet_name}]\n{text}\n")

        return "\n".join(text_data)
    

class GenericFileProcessor(FileProcessor):
    def process_file(self, file_path) -> list[Element]:
        return super().chunks_from_file(file_path)
    
    def docx_ocr_replace(self, input_path, output_path):
        """Replace images in DOCX with OCR text while preserving layout"""

        # Load the document
        doc = Document(input_path)

        # XML namespaces
        namespaces = {
            'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
            'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
            'pic': 'http://schemas.openxmlformats.org/drawingml/2006/picture',
            'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
        }

        # Process each paragraph
        for paragraph in doc.paragraphs:
            # Convert paragraph XML to lxml element for proper namespace handling
            p_xml = paragraph._p.xml
            p_tree = etree.fromstring(p_xml)
            # paragraph.text = ''  # Clear the paragraph text

            # Find all images in this paragraph
            for pic in p_tree.xpath('.//pic:pic', namespaces=namespaces):
                # Get the image relationship ID
                blip = pic.xpath('.//a:blip', namespaces=namespaces)[0]
                rId = blip.attrib.get(f'{{{namespaces["r"]}}}embed')

                if rId and rId in doc.part.related_parts:
                    image_part = doc.part.related_parts[rId]
                    image_data = image_part.blob

                    # Run OCR
                    try:
                        # pyteseract
                        # img = Image.open(io.BytesIO(image_data))
                        # text = pytesseract.image_to_string(img)

                        # unstructured
                        elements = partition(file=io.BytesIO(image_data))
                        text = ''
                        for element in elements:
                            try:
                                text += str(element.text) + '\n'
                            except Exception as e:
                                current_app.logger.error(str(e))

                        # Replace the image with text (simple version - replaces entire paragraph)
                        paragraph.text += text.strip() + '\n'

                        # Alternative: Add text after the image (preserves other content)
                        # paragraph.add_run("\nOCR Result: " + text.strip())

                    except Exception as e:
                        print(f"OCR failed: {str(e)}")
                        
        # Save the modified document
        doc.save(output_path)
    
    def xlsx_ocr_replace(self, input_path, output_path):
        """Replace images in XLSX with OCR text"""
        
        # Load the workbook
        wb = openpyxl.load_workbook(input_path)
        
        for sheet in wb.worksheets:
            to_remove = []
            for image in sheet._images:
                # Extract image data
                img_data = image._data()

                # pytesseract
                # img = Image.open(io.BytesIO(img_data))
                # # Perform OCR
                # text = pytesseract.image_to_string(img)
                
                # unstructured
                elements = partition(file=io.BytesIO(img_data))
                text = ''
                for element in elements:
                    try:
                        text += str(element.text) + '\n'
                    except Exception as e:
                        current_app.logger.error(str(e))
                
                # Remove the image
                to_remove.append(image)
                # Add text to the nearest cell
                cell = sheet.cell(row=image.anchor._from.row+1, 
                                column=image.anchor._from.col+1)
                cell.value = text.strip()
                
            for image_to_remove in to_remove:
                try:
                    sheet._images.remove(image_to_remove)
                except Exception as e:
                    current_app.logger.error(str(e))
        
        wb.save(output_path)
    
    def pptx_ocr_replace(self, input_path, output_path):
        """Replace images in PPTX with OCR text"""
        
        prs = Presentation(input_path)
        # pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Linux path
        
        for slide in prs.slides:
            for shape in list(slide.shapes):  # Create a copy for iteration
                if shape.shape_type == 13:  # Picture type
                    # Extract image
                    img_bytes = shape.image.blob

                    # pytesseract
                    # img = Image.open(io.BytesIO(img_bytes))
                    # # Perform OCR
                    # text = pytesseract.image_to_string(img)

                    # unstructured
                    elements = partition(file=io.BytesIO(img_bytes))
                    text = ''
                    for element in elements:
                        try:
                            text += str(element.text) + '\n'
                        except Exception as e:
                            current_app.logger.error(str(e))
                    
                    # Replace with text box
                    left = shape.left
                    top = shape.top
                    width = shape.width
                    height = shape.height
                    
                    textbox = slide.shapes.add_textbox(left, top, width, height)
                    text_frame = textbox.text_frame
                    text_frame.text = text.strip()
                    
                    # Remove original image
                    slide.shapes._spTree.remove(shape._element)
        
        prs.save(output_path)
    

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

        