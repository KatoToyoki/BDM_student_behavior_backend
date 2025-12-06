"""
Configuration module for the Behavior Analysis application.

Contains all configuration settings for data paths, Spark parameters,
and conversion options.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _detect_environment() -> str:
    """Detect execution environment (docker cluster, docker local, or local)."""
    # Check if running in Docker with Spark cluster
    if os.environ.get("SPARK_MASTER"):
        return "docker_cluster"
    # Check if running in Docker Jupyter
    if Path("/home/jovyan").exists():
        return "docker_local"
    return "local"


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


def _get_spark_master() -> str:
    """Get appropriate Spark master URL based on environment."""
    env = _detect_environment()
    if env == "docker_cluster":
        # Use Spark cluster master from environment variable
        return os.environ.get("SPARK_MASTER", "spark://spark-master:7077")
    elif env == "docker_local":
        # Use local mode in Docker
        return "local[*]"
    else:
        # Use local mode on host machine
        return "local[4]"


@dataclass
class SparkConfig:
    """Configuration for Spark session parameters."""

    # Spark memory configuration (optimized for 19GB RAM environment)
    DRIVER_MEMORY: str = "2g"
    EXECUTOR_MEMORY: str = "2g"
    EXECUTOR_CORES: int = 2

    # Spark optimization parameters
    SQL_SHUFFLE_PARTITIONS: int = 8  # Adjusted for cluster
    MAX_RESULT_SIZE: str = "1g"

    # Spark features
    ADAPTIVE_ENABLED: bool = True
    ADAPTIVE_COALESCE_PARTITIONS: bool = True

    # Serialization
    SERIALIZER: str = "org.apache.spark.serializer.KryoSerializer"

    # Master configuration (auto-detected based on environment)
    MASTER: str = field(default_factory=_get_spark_master)
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

    def get_artifact_path(self, artifact_type: str = "logs") -> str:
        """
        Get full path to artifacts directory (logs, visualizations, etc.).

        Args:
            artifact_type: Type of artifact (logs, visualizations, etc.)

        Returns:
            Full path to the artifact directory
        """
        artifact_dir = Path(self.data.LOG_DIR).parent / artifact_type
        return str(artifact_dir)


# Global configuration instance cache
class _ConfigCache:  # noqa: N801
    """Cache for application configuration."""

    _instance: AppConfig | None = None

    @classmethod
    def get(cls) -> AppConfig:
        """Get or create the global configuration instance."""
        if cls._instance is None:
            cls._instance = AppConfig()
        return cls._instance


def get_config() -> AppConfig:
    """Get the global configuration instance (singleton pattern)."""
    return _ConfigCache.get()


def load_config() -> AppConfig:
    """
    Load configuration from environment variables (if any) and return config instance.

    Environment variables can override default values:
    - BA_DATA_DIR: Override data directory
    - BA_PARQUET_DIR: Override parquet output directory
    - BA_LOG_DIR: Override log directory
    - BA_CHUNK_SIZE: Override conversion chunk size
    - BA_COMPRESSION: Override parquet compression algorithm
    - SPARK_MASTER: Override Spark master URL (for cluster mode)

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

    # Refresh Spark master based on updated environment
    config.spark.MASTER = _get_spark_master()

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
