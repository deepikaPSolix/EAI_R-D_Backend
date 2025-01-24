def delete_files_in_directory(directory_path):
    import os
    try:
        # Loop through all items in the directory
        for item in os.listdir(directory_path):
            # Create full path
            item_path = os.path.join(directory_path, item)
            
            # Check if it's a file (not a directory)
            if os.path.isfile(item_path):
                os.remove(item_path)
                print(f"Deleted: {item_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def delete_files(file_paths: list):
    import os
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                current_app.logger.info(f"Deleted: {file_path}")
        except Exception as e:
            from flask import current_app
            current_app.logger.error(str(e))


def is_audio_or_video_file(file_path):
    """
    Check if a file is an audio or video file based on its extension.

    :param file_path: Path to the file.
    :return: True if the file is audio or video, False otherwise.
    """

    import os
    # Supported audio and video extensions
    audio_extensions = {'.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a', '.wma', '.alac'}
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.webm', '.mpeg', '.3gp'}

    # Extract file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    return ext in audio_extensions or ext in video_extensions