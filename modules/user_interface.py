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
	selected: str = input(
		"Enter the numbers of the folders to select, separated by commas (or type 'q' to exit): ").strip()
	if selected.lower() in ["q", "exit"]:
		print("Exiting.")
		exit(0)
	try:
		indices: List[int] = [int(i.strip()) - 1 for i in selected.split(',') if
		                      i.strip().isdigit()]
		selected_folders: List[Path] = [folders[i] for i in indices if
		                                0 <= i < len(folders)]
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
		logger.info(
			f"No files with extension '{extension}' found in {directory}.")
		return []
	print(f"Files with extension '{extension}' found in {directory}:")
	for idx, file in enumerate(files, 1):
		print(f"{idx}. {file.relative_to(directory)}")
	selected: str = input(
		"Enter the numbers of the files to select, separated by commas (or type 'q' to exit): ").strip()
	if selected.lower() in ["q", "exit"]:
		print("Exiting.")
		exit(0)
	try:
		indices: List[int] = [int(i.strip()) - 1 for i in selected.split(',') if
		                      i.strip().isdigit()]
		selected_files: List[Path] = [files[i] for i in indices if
		                              0 <= i < len(files)]
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
	Prompt user to choose global chunking method or file-by-file selection.
	"""
	choice = input(
		"\nChunking strategy selection:\n"
		f"  [y] Use '{default_method}' method for all files\n"
		"  [n] Choose chunking method individually for each file\n> "
	).strip().lower()

	if choice in ["y", "yes"]:
		print(f"Using '{default_method}' chunking method for all files.")
		return default_method
	else:
		return None


def ask_additional_context_mode() -> dict:
    """
    Prompt the user for how to handle additional context.
    """
    use_context = input(
        "Do you want to enhance extraction with additional context? (y/n):\n"
        "  Adding additional context can significantly improve extraction accuracy for complex documents.\n> "
    ).strip().lower()

    if use_context not in ["y", "yes"]:
        return {
            "use_additional_context": False,
            "use_default_context": False
        }

    context_mode = input(
        "Choose context source:\n"
        "  [s] Use schema-specific default context (pre-configured for this document type)\n"
        "  [f] Use file-specific context files (from {filename}_context.txt files)\n> "
    ).strip().lower()

    if context_mode in ["s"]:
        return {
            "use_additional_context": True,
            "use_default_context": True
        }
    else:
        return {
            "use_additional_context": True,
            "use_default_context": False
        }


def ask_file_chunking_method(file_name: str) -> str:
	"""
	Prompt the user to select a chunking method for the given file.
	"""
	print(f"\nSelect chunking method for file '{file_name}':")
	print(
		"  1. Automatic chunking - Split text based on token limits with no intervention")
	print(
		"  2. Interactive chunking - View default chunks and manually adjust boundaries")
	print(
		"  3. Predefined chunks - Use saved boundaries from {file}_line_ranges.txt file")

	choice = input("Enter option (1-3): ").strip()
	if choice == "1":
		return "auto"
	elif choice == "2":
		return "auto-adjust"
	elif choice == "3":
		return "line_ranges.txt"
	else:
		print("Invalid selection, defaulting to automatic chunking.")
		return "auto"


def ask_file_additional_context(file_name: str) -> Optional[str]:
	"""
    Prompt the user to provide custom additional context for a specific file.

    Parameters:
    - file_name (str): The name of the file being processed.

    Returns:
    - Optional[str]: The custom additional context text or None if not provided
    """
	print(f"Enter custom additional context for file '{file_name}':")
	print(
		"(Enter text, then press Enter twice or type '\\done' on a new line to finish)")

	lines = []
	while True:
		line = input()
		if not line or line.strip() == "\\done":
			break
		lines.append(line)

	context = "\n".join(lines).strip()
	return context if context else None
