"""
Data validation module.

Provides validation functions for SPSS and Parquet files to ensure
conversion integrity.
"""

import os
from typing import Tuple, Optional

import pyreadstat
import pyarrow.parquet as pq

from ..utils.logger import get_logger
from ..utils.file_utils import check_disk_space, estimate_parquet_size


class DataValidator:
    """Validator for SPSS and Parquet data files."""

    def __init__(self):
        """Initialize the validator."""
        self.logger = get_logger()

    def validate_spss_file(self, spss_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SPSS file before conversion.

        Args:
            spss_path: Path to SPSS file

        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        # Check file exists
        if not os.path.exists(spss_path):
            return False, f"File not found: {spss_path}"

        # Check file is readable
        if not os.access(spss_path, os.R_OK):
            return False, f"File not readable: {spss_path}"

        # Try to read metadata
        try:
            _, meta = pyreadstat.read_sav(spss_path, metadataonly=True)

            if meta.number_rows == 0:
                return False, "SPSS file contains no rows"

            if meta.number_columns == 0:
                return False, "SPSS file contains no columns"

            self.logger.info(
                f"SPSS validation passed: {meta.number_rows:,} rows, "
                f"{meta.number_columns} columns"
            )
            return True, None

        except Exception as e:
            return False, f"Failed to read SPSS file: {e}"

    def validate_disk_space(
        self,
        spss_path: str,
        output_dir: str,
        compression: str = "snappy"
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate sufficient disk space for conversion.

        Args:
            spss_path: Path to SPSS file
            output_dir: Output directory for Parquet file
            compression: Compression algorithm

        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        spss_size = os.path.getsize(spss_path)
        estimated_parquet_size = estimate_parquet_size(spss_size, compression)

        # Add 20% buffer for safety
        required_space = int(estimated_parquet_size * 1.2)

        if not check_disk_space(output_dir, required_space):
            required_gb = required_space / (1024 ** 3)
            return False, f"Insufficient disk space. Required: {required_gb:.2f} GB"

        return True, None

    def validate_conversion(
        self,
        spss_path: str,
        parquet_path: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate conversion result.

        Checks:
        - Parquet file exists
        - Row count matches
        - Column count matches

        Args:
            spss_path: Path to source SPSS file
            parquet_path: Path to generated Parquet file

        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        # Check Parquet file exists
        if not os.path.exists(parquet_path):
            return False, f"Parquet file not found: {parquet_path}"

        try:
            # Get SPSS metadata
            _, spss_meta = pyreadstat.read_sav(spss_path, metadataonly=True)
            spss_rows = spss_meta.number_rows
            spss_cols = spss_meta.number_columns

            # Get Parquet metadata
            parquet_file = pq.ParquetFile(parquet_path)
            parquet_rows = parquet_file.metadata.num_rows
            parquet_cols = len(parquet_file.schema)

            # Validate row count
            if spss_rows != parquet_rows:
                return False, (
                    f"Row count mismatch: SPSS has {spss_rows:,} rows, "
                    f"Parquet has {parquet_rows:,} rows"
                )

            # Validate column count
            if spss_cols != parquet_cols:
                return False, (
                    f"Column count mismatch: SPSS has {spss_cols} columns, "
                    f"Parquet has {parquet_cols} columns"
                )

            self.logger.info(
                f"Conversion validation passed: {parquet_rows:,} rows, "
                f"{parquet_cols} columns"
            )
            return True, None

        except Exception as e:
            return False, f"Validation failed: {e}"

    def validate_all(
        self,
        spss_path: str,
        parquet_path: str,
        output_dir: str,
        compression: str = "snappy"
    ) -> Tuple[bool, Optional[str]]:
        """
        Run all validation checks.

        Args:
            spss_path: Path to SPSS file
            parquet_path: Path to Parquet file
            output_dir: Output directory
            compression: Compression algorithm

        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        # Validate SPSS file
        valid, error = self.validate_spss_file(spss_path)
        if not valid:
            return False, f"SPSS validation failed: {error}"

        # Validate disk space
        valid, error = self.validate_disk_space(spss_path, output_dir, compression)
        if not valid:
            return False, f"Disk space validation failed: {error}"

        # If Parquet exists, validate conversion
        if os.path.exists(parquet_path):
            valid, error = self.validate_conversion(spss_path, parquet_path)
            if not valid:
                return False, f"Conversion validation failed: {error}"

        return True, None


def validate_spss_file(spss_path: str) -> Tuple[bool, Optional[str]]:
    """
    Convenience function to validate SPSS file.

    Args:
        spss_path: Path to SPSS file

    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    validator = DataValidator()
    return validator.validate_spss_file(spss_path)


def validate_conversion(spss_path: str, parquet_path: str) -> Tuple[bool, Optional[str]]:
    """
    Convenience function to validate conversion.

    Args:
        spss_path: Path to SPSS file
        parquet_path: Path to Parquet file

    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    validator = DataValidator()
    return validator.validate_conversion(spss_path, parquet_path)
