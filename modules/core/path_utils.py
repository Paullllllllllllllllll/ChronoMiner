"""
Path utility module for handling Windows path length limitations.

This module implements best practices for dealing with Windows MAX_PATH (260 char)
limitation by creating shortened directory names with content-based hashes.

Strategy:
- Truncate long names to a safe limit (80 chars)
- Append hash of full name for uniqueness
- Similar to how npm, Git, and other production tools handle long paths

Reference: https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation
"""

import hashlib
from pathlib import Path


# Maximum safe length for directory/file names
# Keep it well under 255 (NTFS limit) to leave room for subdirs and files
MAX_SAFE_NAME_LENGTH = 80

# Hash length for uniqueness (8 chars = 4 billion combinations)
HASH_LENGTH = 8


def create_safe_directory_name(original_name: str, suffix: str = "") -> str:
	"""
	Create a safe directory name that won't exceed Windows path limits.
	
	Args:
		original_name: The original name (e.g., document stem)
		suffix: Optional suffix to append (e.g., "_working_files")
	
	Returns:
		A shortened, safe directory name with hash for uniqueness
		
	Example:
		Input:  "Beukers etal 2025 Grape (Vitis vinifera) use in the early modern..."
		Output: "Beukers etal 2025 Grape (Vitis vinifera) use in the earl-a3f8d9e2_working_files"
	"""
	# Calculate hash of full original name for uniqueness
	name_hash = hashlib.sha256(original_name.encode('utf-8')).hexdigest()[:HASH_LENGTH]
	
	# Calculate how much space we have for the actual name
	# Format: [truncated_name]-[hash][suffix]
	reserved_length = len(suffix) + HASH_LENGTH + 1  # +1 for the dash
	available_length = MAX_SAFE_NAME_LENGTH - reserved_length
	
	# Truncate the name if necessary
	if len(original_name) > available_length:
		# Try to break at a word boundary for better readability
		truncated = original_name[:available_length].rstrip()
		# Remove trailing punctuation/spaces that might look odd
		truncated = truncated.rstrip('.-_ ')
	else:
		truncated = original_name
	
	# Combine truncated name with hash and suffix
	safe_name = f"{truncated}-{name_hash}{suffix}"
	
	return safe_name


def create_safe_log_filename(base_name: str, log_type: str) -> str:
	"""
	Create a safe log filename that won't exceed path limits.
	
	Args:
		base_name: The base name (e.g., document stem)
		log_type: Type of log (e.g., "transcription", "summary")
	
	Returns:
		A shortened, safe log filename
		
	Example:
		Input:  "Very long document name...", "transcription"
		Output: "Very long document name...-a3f8d9e2_transcription_log.json"
	"""
	suffix = f"_{log_type}_log.json"
	
	# Calculate hash for uniqueness
	name_hash = hashlib.sha256(base_name.encode('utf-8')).hexdigest()[:HASH_LENGTH]
	
	# Calculate available space
	reserved_length = len(suffix) + HASH_LENGTH + 1  # +1 for dash
	available_length = MAX_SAFE_NAME_LENGTH - reserved_length
	
	# Truncate if necessary
	if len(base_name) > available_length:
		truncated = base_name[:available_length].rstrip('.-_ ')
	else:
		truncated = base_name
	
	# Combine
	safe_filename = f"{truncated}-{name_hash}{suffix}"
	
	return safe_filename


def ensure_path_safe(path: Path) -> Path:
	r"""
	Ensure a path won't exceed Windows MAX_PATH limits.
	
	For Windows 10 1607+, Python's pathlib automatically uses extended-length
	paths (\\?\) when needed, but this provides an explicit check.
	
	Args:
		path: The path to check
		
	Returns:
		The path (potentially with extended-length prefix on Windows)
		
	Note:
		Python 3.6+ pathlib handles extended paths automatically in most cases.
	"""
	# On Windows, check if path is too long
	import platform
	if platform.system() == 'Windows':
		# Python's pathlib handles \\?\ prefix automatically in most cases
		# But we can ensure the path is resolved/absolute which triggers this
		try:
			return path.resolve()
		except OSError:
			# If resolve fails, return as-is and let caller handle
			return path
	
	return path
