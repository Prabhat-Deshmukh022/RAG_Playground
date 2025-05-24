import os
import shutil

def clear_directory(path: str):
    if not os.path.isdir(path):
        raise ValueError(f"'{path}' is not a valid directory.")

    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # Delete file or symbolic link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Recursively delete subdirectory
        except Exception as e:
            print(f"Failed to delete '{file_path}': {e}")
