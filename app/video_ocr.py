
import cv2
import numpy as np
import re
from paddleocr import PaddleOCR
import difflib
import moviepy as mp

import numpy as np

class VideoOcr:
    
    @staticmethod
    def extract_text_from_video(video_path:str, desired_fps:int=1, min_similarity_threshold:float=0.7, pixel_diff_threshold = 0.5) -> str:
        """
        Extract text from video frames using PaddleOCR with pixel difference threshold.

        Args:
            video_path (str): Path to the video file.
            desired_fps (int): Number of frames to process per second (default: 1 frame per second).
            min_similarity_threshold (float): Minimum threshold to detect significant text change between frames.

        Returns:
            list: A list of unique text entries in the order they appear in the video.
        """

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Unable to open video file {video_path}")
            return []

        # Get the video's FPS dynamically
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print("Error: Unable to retrieve FPS from video. Check the video file.")
            return []

        # Calculate total frame count (First Pass)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Total Frame Count:", total_frames)

        # Text processing (Second Pass)
        text_data = []
        frame_count = 0
        processed_frames = 0
        next_frame = 0
        previous_text = ""
        previous_frame = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process only the desired FPS frames
            if frame_count == next_frame:
                next_frame += round(fps / desired_fps)

                # Pixel difference check
                if previous_frame is not None:
                    pixel_diff = VideoOcr._calculate_pixel_difference(previous_frame, frame)
                    if pixel_diff < pixel_diff_threshold:
                        frame_count += 1
                        continue  # Skip processing if difference is below the threshold

                processed_frames += 1

                # Extract text from the current frame
                current_text = VideoOcr._process_frame(frame)
                if current_text and not VideoOcr._is_similar(current_text, previous_text, min_similarity_threshold):
                    text_data.append((frame_count, current_text))
                    previous_text = current_text

                previous_frame = frame  # Update the previous frame reference

            frame_count += 1

        cap.release()

        print("Processed Frame Count:", processed_frames)
        print("Text extraction complete.")

        # Sort text by frame number and return the ordered list
        sorted_text_data = sorted(text_data, key=lambda x: x[0])
        final_text = [text for _, text in sorted_text_data]

        return "".join(final_text)
        
    @staticmethod
    def _process_frame(frame):
        """
        Process a frame for OCR and extract text.

        Args:
            frame (numpy.ndarray): The current frame from the video.

        Returns:
            str: Extracted text from the frame.
        """
        try:
            ocr = PaddleOCR(use_angle_cls=True, lang='en')
            result = ocr.ocr(frame, det=True, cls=True)
            if result and len(result) > 0:
                text = [item[1][0] for item in result[0] if len(item) > 1 and item[1]]
                return ' '.join(text)
            else:
                return ""
        except Exception as e:
            print(f"Error during OCR processing: {e}")
            return ""
        

    @staticmethod
    def _calculate_pixel_difference(prev_frame, curr_frame):
        """
        Calculate the pixel difference between two frames.

        Args:
            prev_frame (numpy.ndarray): The previous frame.
            curr_frame (numpy.ndarray): The current frame.

        Returns:
            float: The percentage of pixel difference.
        """
        diff = cv2.absdiff(prev_frame, curr_frame)
        non_zero_diff = np.count_nonzero(diff)
        total_pixels = np.prod(diff.shape)  # Total number of pixels in the frame
        return non_zero_diff / total_pixels  # Return the fraction of different pixels

    @staticmethod
    def _is_similar(current_text, previous_text, threshold=0.7):
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
        current_text = VideoOcr._normalize_text(current_text)
        previous_text = VideoOcr._normalize_text(previous_text)
        similarity_ratio = difflib.SequenceMatcher(None, current_text, previous_text).ratio()
        return similarity_ratio > threshold

    @staticmethod
    def _normalize_text(text):
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