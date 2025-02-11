# modules/user_interface.py

from pathlib import Path
from typing import List
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
