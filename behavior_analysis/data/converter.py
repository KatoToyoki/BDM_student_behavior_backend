"""
SPSS to Parquet converter module.

Handles conversion of SPSS (.sav) files to Parquet format with support for
large files through batched processing.
"""

import os
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pyreadstat

from ..config import ConversionConfig
from ..utils.file_utils import (
    check_disk_space,
    ensure_directory_exists,
    estimate_parquet_size,
    get_file_size,
    mark_conversion_completed,
    should_reconvert,
)
from ..utils.logger import get_logger


class SPSSToParquetConverter:
    """
    Converter for transforming SPSS files to Parquet format.

    Supports both small files (loaded entirely into memory) and large files
    (processed in batches to avoid memory issues).
    """

    def __init__(self, config: ConversionConfig | None = None):
        """
        Initialize the converter.

        Args:
            config: Conversion configuration. If None, uses default config.
        """
        from ..config import get_conversion_config

        self.config = config if config is not None else get_conversion_config()
        self.logger = get_logger()

    def is_converted(self, parquet_path: str, spss_path: str) -> bool:
        """
        Check if SPSS file has already been converted.

        Args:
            parquet_path: Path to parquet file
            spss_path: Path to SPSS file

        Returns:
            bool: True if already converted and up-to-date
        """
        return not should_reconvert(parquet_path, spss_path)

    def convert_file(self, spss_path: str, parquet_path: str, force: bool = False) -> bool:
        """
        Convert SPSS file to Parquet format.

        Args:
            spss_path: Path to input SPSS file
            parquet_path: Path to output Parquet file
            force: If True, reconvert even if already converted

        Returns:
            bool: True if conversion succeeded

        Raises:
            FileNotFoundError: If SPSS file doesn't exist
            IOError: If disk space insufficient or other IO error
            pyreadstat.ReadstatError: If SPSS file cannot be read
        """
        # Validate input file
        spss = Path(spss_path)
        if not spss.exists():
            raise FileNotFoundError(f"SPSS file not found: {spss_path}")

        # Check if already converted
        if not force and self.is_converted(parquet_path, spss_path):
            self.logger.info(f"File already converted: {parquet_path}")
            return True

        # Get file size and determine strategy
        spss_size = spss.stat().st_size
        spss_size_hr = get_file_size(spss_path)
        self.logger.info(f"Converting SPSS file: {spss_path} ({spss_size_hr})")

        # Ensure output directory exists
        parquet = Path(parquet_path)
        parquet_dir = str(parquet.parent)
        ensure_directory_exists(parquet_dir)

        # Check disk space
        estimated_size = estimate_parquet_size(spss_size, self.config.COMPRESSION)
        if not check_disk_space(parquet_dir, estimated_size):
            raise OSError(f"Insufficient disk space. Required: {estimated_size / (1024**3):.2f} GB")

        # Choose conversion strategy based on file size
        try:
            if spss_size < self.config.LARGE_FILE_THRESHOLD:
                self.logger.info("Using small file conversion strategy")
                success = self._convert_small_file(spss_path, parquet_path)
            else:
                self.logger.info(
                    f"Using large file conversion strategy (batch size: {self.config.CHUNK_SIZE:,} rows)"
                )
                success = self._convert_large_file(spss_path, parquet_path)

            if success:
                # Mark as converted
                _, meta = pyreadstat.read_sav(spss_path, metadataonly=True)
                mark_conversion_completed(
                    parquet_path,
                    spss_path,
                    metadata={
                        "row_count": meta.number_rows,
                        "column_count": meta.number_columns,
                        "compression": self.config.COMPRESSION,
                    },
                )

                result_size = get_file_size(parquet_path)
                self.logger.info(f"Conversion completed: {parquet_path} ({result_size})")

            return success

        except Exception as e:
            self.logger.error(f"Conversion failed: {e}")
            # Clean up partial file if it exists
            parquet_file = Path(parquet_path)
            if parquet_file.exists():
                try:
                    parquet_file.unlink()
                except OSError:
                    pass
            raise

    def _convert_small_file(self, spss_path: str, parquet_path: str) -> bool:
        """
        Convert small SPSS file by loading entirely into memory.

        Args:
            spss_path: Path to SPSS file
            parquet_path: Path to output Parquet file

        Returns:
            bool: True if successful
        """
        # Read entire SPSS file
        self.logger.info("Reading SPSS file into memory...")
        df, meta = pyreadstat.read_sav(spss_path, user_missing=True)

        self.logger.info(f"Loaded {meta.number_rows:,} rows, {meta.number_columns} columns")

        # Convert to PyArrow Table
        self.logger.info("Converting to Parquet format...")
        table = pa.Table.from_pandas(df)

        # Write to Parquet
        pq.write_table(table, parquet_path, compression=self.config.COMPRESSION)

        return True

    def _convert_large_file(self, spss_path: str, parquet_path: str) -> bool:
        """
        Convert large SPSS file using batched reading.

        Args:
            spss_path: Path to SPSS file
            parquet_path: Path to output Parquet file

        Returns:
            bool: True if successful
        """
        # Get metadata first
        self.logger.info("Reading SPSS metadata...")
        _, meta = pyreadstat.read_sav(spss_path, metadataonly=True)
        total_rows = meta.number_rows

        self.logger.info(
            f"Total rows: {total_rows:,}, processing in batches of {self.config.CHUNK_SIZE:,}"
        )

        # Use temporary file to ensure atomic write
        temp_dir = str(Path(parquet_path).parent)
        temp_fd, temp_path = tempfile.mkstemp(suffix=".parquet", dir=temp_dir)
        os.close(temp_fd)

        writer = None
        batch_count = 0

        try:
            # Process in batches
            for offset in range(0, total_rows, self.config.CHUNK_SIZE):
                batch_count += 1
                rows_to_read = min(self.config.CHUNK_SIZE, total_rows - offset)

                self.logger.info(
                    f"Processing batch {batch_count}: rows {offset:,} to {offset + rows_to_read:,}"
                )

                # Read batch
                df, _ = pyreadstat.read_sav(
                    spss_path, row_offset=offset, row_limit=rows_to_read, user_missing=True
                )

                # Convert to PyArrow Table
                table = pa.Table.from_pandas(df)

                # Initialize writer on first batch
                if writer is None:
                    writer = pq.ParquetWriter(
                        temp_path, table.schema, compression=self.config.COMPRESSION
                    )

                # Write batch
                writer.write_table(table)

                # Log progress
                progress = (offset + rows_to_read) / total_rows * 100
                self.logger.info(f"Progress: {progress:.1f}%")

            # Close writer
            if writer:
                writer.close()

            # Move temp file to final location (atomic operation)
            Path(temp_path).replace(parquet_path)

            self.logger.info(f"Successfully processed {batch_count} batches")
            return True

        except Exception:
            # Clean up on error
            if writer:
                writer.close()
            temp_file = Path(temp_path)
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            raise

    def convert_all(
        self, spss_files: dict[str, str], data_dir: str, parquet_dir: str, force: bool = False
    ) -> dict[str, dict[str, bool | str | None]]:
        """
        Convert multiple SPSS files to Parquet.

        Args:
            spss_files: Dictionary mapping dataset names to SPSS filenames
            data_dir: Directory containing SPSS files
            parquet_dir: Directory for output Parquet files
            force: If True, reconvert even if already converted

        Returns:
            dict: Results with dataset names as keys and success status as values
        """
        results: dict[str, dict[str, bool | str | None]] = {}
        data_path = Path(data_dir)
        parquet_path_obj = Path(parquet_dir)

        for dataset_name, spss_filename in spss_files.items():
            spss_path = str(data_path / spss_filename)
            parquet_path = str(parquet_path_obj / f"{dataset_name}.parquet")

            self.logger.info(f"Processing dataset: {dataset_name}")

            try:
                success = self.convert_file(spss_path, parquet_path, force=force)
                results[dataset_name] = {"success": success, "error": None}
            except Exception as e:
                self.logger.error(f"Failed to convert {dataset_name}: {e}")
                results[dataset_name] = {"success": False, "error": str(e)}

        return results
