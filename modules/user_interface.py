# modules/user_interface.py

from pathlib import Path
from typing import List, Optional
from modules.logger import setup_logger

logger = setup_logger(__name__)


def select_folders(directory: Path) -> List[Path]:
    """
    Prompt the user to select folders from the given directory.

    :param directory: Directory to search for folders.
    :return: List of selected folder paths.
    """
    folders: List[Path] = [d for d in directory.iterdir() if d.is_dir()]
    if not folders:
        print(f"No folders found in {directory}.")
        logger.info(f"No folders found in {directory}.")
        return []
    print(f"Folders found in {directory}:")
    for idx, folder in enumerate(folders, 1):
        print(f"{idx}. {folder.name}")
    selected: str = input("Enter the numbers of the folders to select, separated by commas (or type 'q' to exit): ").strip()
    if selected.lower() in ["q", "exit"]:
        print("Exiting.")
        exit(0)
    try:
        indices: List[int] = [int(i.strip()) - 1 for i in selected.split(',') if i.strip().isdigit()]
        selected_folders: List[Path] = [folders[i] for i in indices if 0 <= i < len(folders)]
        if not selected_folders:
            print("No valid folders selected.")
            logger.info("No valid folders selected by the user.")
        return selected_folders
    except ValueError:
        print("Invalid input. Please enter numbers separated by commas.")
        logger.error("User entered invalid folder selection input.")
        return []


def select_files(directory: Path, extension: str) -> List[Path]:
    """
    Prompt the user to select files with a given extension from the specified directory.

    :param directory: Directory to search for files.
    :param extension: File extension to filter (e.g., '.txt').
    :return: List of selected file paths.
    """
    files: List[Path] = list(directory.rglob(f"*{extension.lower()}"))
    if not files:
        print(f"No files with extension '{extension}' found in {directory}.")
        logger.info(f"No files with extension '{extension}' found in {directory}.")
        return []
    print(f"Files with extension '{extension}' found in {directory}:")
    for idx, file in enumerate(files, 1):
        print(f"{idx}. {file.relative_to(directory)}")
    selected: str = input("Enter the numbers of the files to select, separated by commas (or type 'q' to exit): ").strip()
    if selected.lower() in ["q", "exit"]:
        print("Exiting.")
        exit(0)
    try:
        indices: List[int] = [int(i.strip()) - 1 for i in selected.split(',') if i.strip().isdigit()]
        selected_files: List[Path] = [files[i] for i in indices if 0 <= i < len(files)]
        if not selected_files:
            print("No valid files selected.")
            logger.info("No valid files selected by the user.")
        return selected_files
    except ValueError:
        print("Invalid input. Please enter numbers separated by commas.")
        logger.error("User entered invalid file selection input.")
        return []


def ask_global_chunking_mode(default_method: str) -> Optional[str]:
    """
    Prompt the user whether to use the default chunking method for all files.
    If the user chooses yes, return the default_method; otherwise, return None.

    Parameters:
    - default_method (str): The default chunking method from chunking_config.yaml.

    Returns:
    - Optional[str]: The chosen global chunking method, or None if the user opts for manual selection.
    """
    choice = input(
        "Do you want to use the default chunking method for all files? (y/n): ").strip().lower()
    if choice in ["y", "yes"]:
        print(f"Using default chunking method: {default_method}")
        return default_method
    else:
        return None


def ask_file_chunking_method(file_name: str) -> str:
    """
    Prompt the user to select a chunking method for the given file.

    Parameters:
    - file_name (str): The name of the file being processed.

    Returns:
    - str: The chosen chunking method. One of: "auto", "auto-adjust", "line_ranges.txt".
    """
    print(f"Select chunking method for file '{file_name}':")
    print("1. Automatic token-based chunking (auto)")
    print(
        "2. Automatic token-based chunking with manual re-adjustments (auto-adjust)")
    print("3. Use _line_ranges.txt file (line_ranges.txt)")
    choice = input("Enter 1, 2, or 3: ").strip()
    if choice == "1":
        return "auto"
    elif choice == "2":
        return "auto-adjust"
    elif choice == "3":
        return "line_ranges.txt"
    else:
        print("Invalid selection, defaulting to 'auto'.")
        return "auto"
