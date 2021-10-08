import os.path


def check_file_exists(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Provided file {file_path} doesn't exist.")

    return file_path
