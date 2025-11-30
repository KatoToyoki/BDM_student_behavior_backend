"""
File utility functions for the application.

Provides common file operations like directory creation, disk space checking,
and conversion status tracking.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


def ensure_directory_exists(directory_path: str) -> None:
    """
    Create directory if it doesn't exist.

    Args:
        directory_path: Path to directory to create

    Raises:
        OSError: If directory creation fails
    """
    os.makedirs(directory_path, exist_ok=True)


def check_disk_space(path: str, required_bytes: int) -> bool:
    """
    Check if there's enough disk space available.

    Args:
        path: Path to check disk space for
        required_bytes: Required space in bytes

    Returns:
        bool: True if enough space available, False otherwise
    """
    stat = shutil.disk_usage(path)
    return stat.free >= required_bytes


def get_file_size(file_path: str, human_readable: bool = True) -> str:
    """
    Get file size in human-readable format or bytes.

    Args:
        file_path: Path to file
        human_readable: If True, return formatted string (e.g., "1.5 GB")

    Returns:
        str: File size as string

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    size_bytes = os.path.getsize(file_path)

    if not human_readable:
        return str(size_bytes)

    # Convert to human-readable format
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0

    return f"{size_bytes:.2f} PB"


def get_conversion_marker_path(parquet_path: str) -> str:
    """
    Get path to conversion marker file.

    Args:
        parquet_path: Path to parquet file

    Returns:
        str: Path to marker file
    """
    parquet_dir = os.path.dirname(parquet_path)
    parquet_name = os.path.basename(parquet_path)
    return os.path.join(parquet_dir, f".{parquet_name}.converted")


def is_conversion_completed(parquet_path: str) -> bool:
    """
    Check if conversion has been completed for a given parquet file.

    Args:
        parquet_path: Path to parquet file to check

    Returns:
        bool: True if conversion marker exists and parquet file exists
    """
    marker_path = get_conversion_marker_path(parquet_path)

    # Check if both marker file and parquet file exist
    if not os.path.exists(marker_path):
        return False

    if not os.path.exists(parquet_path):
        # Marker exists but parquet doesn't - cleanup marker
        try:
            os.remove(marker_path)
        except OSError:
            pass
        return False

    return True


def mark_conversion_completed(
    parquet_path: str,
    spss_path: str,
    metadata: Optional[Dict] = None
) -> None:
    """
    Mark conversion as completed by creating a marker file.

    Args:
        parquet_path: Path to generated parquet file
        spss_path: Path to source SPSS file
        metadata: Optional metadata to store (e.g., row count, column count)

    Raises:
        OSError: If marker file creation fails
    """
    marker_path = get_conversion_marker_path(parquet_path)

    # Prepare conversion metadata
    conversion_info = {
        "timestamp": datetime.now().isoformat(),
        "spss_path": spss_path,
        "parquet_path": parquet_path,
        "spss_size": os.path.getsize(spss_path),
        "parquet_size": os.path.getsize(parquet_path),
        "spss_mtime": os.path.getmtime(spss_path)
    }

    if metadata:
        conversion_info["metadata"] = metadata

    # Write marker file
    with open(marker_path, 'w', encoding='utf-8') as f:
        json.dump(conversion_info, f, indent=2)


def get_conversion_info(parquet_path: str) -> Optional[Dict]:
    """
    Get conversion information from marker file.

    Args:
        parquet_path: Path to parquet file

    Returns:
        Optional[Dict]: Conversion info dict if marker exists, None otherwise
    """
    marker_path = get_conversion_marker_path(parquet_path)

    if not os.path.exists(marker_path):
        return None

    try:
        with open(marker_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def should_reconvert(parquet_path: str, spss_path: str) -> bool:
    """
    Determine if SPSS file should be reconverted.

    Returns True if:
    - Conversion marker doesn't exist
    - Parquet file doesn't exist
    - SPSS file has been modified since last conversion

    Args:
        parquet_path: Path to parquet file
        spss_path: Path to SPSS file

    Returns:
        bool: True if reconversion needed
    """
    # Check if conversion completed
    if not is_conversion_completed(parquet_path):
        return True

    # Get conversion info
    info = get_conversion_info(parquet_path)
    if not info:
        return True

    # Check if SPSS file has been modified
    current_mtime = os.path.getmtime(spss_path)
    previous_mtime = info.get("spss_mtime", 0)

    if current_mtime > previous_mtime:
        return True

    return False


def cleanup_conversion_markers(parquet_dir: str) -> int:
    """
    Clean up orphaned conversion marker files.

    Removes marker files where corresponding parquet file doesn't exist.

    Args:
        parquet_dir: Directory containing parquet files

    Returns:
        int: Number of markers cleaned up
    """
    if not os.path.exists(parquet_dir):
        return 0

    count = 0
    for filename in os.listdir(parquet_dir):
        if filename.startswith('.') and filename.endswith('.converted'):
            marker_path = os.path.join(parquet_dir, filename)

            # Extract parquet filename from marker filename
            # .student.parquet.converted -> student.parquet
            parquet_name = filename[1:-10]  # Remove leading '.' and '.converted'
            parquet_path = os.path.join(parquet_dir, parquet_name)

            if not os.path.exists(parquet_path):
                try:
                    os.remove(marker_path)
                    count += 1
                except OSError:
                    pass

    return count


def estimate_parquet_size(spss_size: int, compression: str = "snappy") -> int:
    """
    Estimate parquet file size based on SPSS file size.

    Args:
        spss_size: Size of SPSS file in bytes
        compression: Compression algorithm (snappy, gzip, etc.)

    Returns:
        int: Estimated parquet size in bytes
    """
    # Compression ratios (approximate)
    compression_ratios = {
        "none": 0.8,      # Slight reduction due to columnar format
        "snappy": 0.5,    # 50-60% compression
        "gzip": 0.35,     # 30-40% compression
        "lz4": 0.6,       # 60-70% compression
        "zstd": 0.4       # 40-50% compression
    }

    ratio = compression_ratios.get(compression.lower(), 0.5)
    return int(spss_size * ratio)
