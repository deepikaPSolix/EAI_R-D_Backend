import os


UPLOAD_FOLDER = 'cache/uploads'
ML_FOLDER = 'cache/ml-model'

def delete_files_in_directory(directory_path):
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