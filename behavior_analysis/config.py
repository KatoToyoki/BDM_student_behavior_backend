"""
Configuration module for the Behavior Analysis application.

Contains all configuration settings for data paths, Spark parameters,
and conversion options.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data paths and files."""

    # Data directories
    DATA_DIR: str = "/data"
    PARQUET_DIR: str = "/data/parquet"
    LOG_DIR: str = "/home/jovyan/workspace/artifacts/logs"

    # SPSS files mapping
    SPSS_FILES: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize SPSS files mapping after dataclass creation."""
        if not self.SPSS_FILES:
            self.SPSS_FILES = {
                "student": "CY08MSP_STU_QQQ.SAV",
                "teacher": "CY08MSP_TCH_QQQ.SAV",
                "school": "CY08MSP_SCH_QQQ.SAV",
            }


@dataclass
class SparkConfig:
    """Configuration for Spark session parameters."""

    # Spark memory configuration (optimized for 19GB RAM environment)
    DRIVER_MEMORY: str = "6g"
    EXECUTOR_MEMORY: str = "8g"
    EXECUTOR_CORES: int = 4

    # Spark optimization parameters
    SQL_SHUFFLE_PARTITIONS: int = 20  # Default 200 is too high for local
    MAX_RESULT_SIZE: str = "2g"

    # Spark features
    ADAPTIVE_ENABLED: bool = True
    ADAPTIVE_COALESCE_PARTITIONS: bool = True

    # Serialization
    SERIALIZER: str = "org.apache.spark.serializer.KryoSerializer"

    # Master configuration
    MASTER: str = "local[4]"  # Use 4 cores, avoid over-parallelization
    APP_NAME: str = "BehaviorAnalysis"


@dataclass
class ConversionConfig:
    """Configuration for SPSS to Parquet conversion."""

    # Batch size for reading large SPSS files
    CHUNK_SIZE: int = 100000  # rows per batch

    # Parquet compression algorithm
    COMPRESSION: str = "snappy"  # Options: snappy, gzip, lz4, zstd

    # File size threshold for determining conversion strategy (in bytes)
    LARGE_FILE_THRESHOLD: int = 500 * 1024 * 1024  # 500 MB

    # Validation settings
    ENABLE_VALIDATION: bool = True
    SAMPLE_VALIDATION_ROWS: int = 100


@dataclass
class AppConfig:
    """Main application configuration combining all config sections."""

    data: DataConfig
    spark: SparkConfig
    conversion: ConversionConfig

    def __init__(self) -> None:
        """Initialize all configuration sections."""
        self.data = DataConfig()
        self.spark = SparkConfig()
        self.conversion = ConversionConfig()

    def get_spss_path(self, dataset_name: str) -> str:
        """Get full path to SPSS file for a given dataset."""
        if dataset_name not in self.data.SPSS_FILES:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return str(Path(self.data.DATA_DIR) / self.data.SPSS_FILES[dataset_name])

    def get_parquet_path(self, dataset_name: str) -> str:
        """Get full path to Parquet file for a given dataset."""
        if dataset_name not in self.data.SPSS_FILES:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return str(Path(self.data.PARQUET_DIR) / f"{dataset_name}.parquet")


# Global configuration instance
_config = None


def get_config() -> AppConfig:
    """Get the global configuration instance (singleton pattern)."""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def load_config() -> AppConfig:
    """
    Load configuration from environment variables (if any) and return config instance.

    Environment variables can override default values:
    - BA_DATA_DIR: Override data directory
    - BA_PARQUET_DIR: Override parquet output directory
    - BA_CHUNK_SIZE: Override conversion chunk size
    - BA_COMPRESSION: Override parquet compression algorithm

    Returns:
        AppConfig: The application configuration instance
    """
    config = get_config()

    # Override with environment variables if present
    if "BA_DATA_DIR" in os.environ:
        config.data.DATA_DIR = os.environ["BA_DATA_DIR"]

    if "BA_PARQUET_DIR" in os.environ:
        config.data.PARQUET_DIR = os.environ["BA_PARQUET_DIR"]

    if "BA_LOG_DIR" in os.environ:
        config.data.LOG_DIR = os.environ["BA_LOG_DIR"]

    if "BA_CHUNK_SIZE" in os.environ:
        config.conversion.CHUNK_SIZE = int(os.environ["BA_CHUNK_SIZE"])

    if "BA_COMPRESSION" in os.environ:
        config.conversion.COMPRESSION = os.environ["BA_COMPRESSION"]

    if "BA_SPARK_DRIVER_MEMORY" in os.environ:
        config.spark.DRIVER_MEMORY = os.environ["BA_SPARK_DRIVER_MEMORY"]

    if "BA_SPARK_EXECUTOR_MEMORY" in os.environ:
        config.spark.EXECUTOR_MEMORY = os.environ["BA_SPARK_EXECUTOR_MEMORY"]

    return config


# Convenience access to configuration
def get_data_config() -> DataConfig:
    """Get data configuration."""
    return get_config().data


def get_spark_config() -> SparkConfig:
    """Get Spark configuration."""
    return get_config().spark


def get_conversion_config() -> ConversionConfig:
    """Get conversion configuration."""
    return get_config().conversion
