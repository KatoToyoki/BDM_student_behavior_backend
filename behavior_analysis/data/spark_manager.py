"""
Spark session management module.

Provides centralized management of Spark sessions with optimized configuration
for local execution.
"""

from typing import Optional

from pyspark.sql import SparkSession

from ..config import SparkConfig
from ..utils.logger import get_logger


class SparkSessionManager:
    """
    Singleton manager for Spark sessions.

    Ensures only one Spark session is active at a time and provides
    context manager support for automatic cleanup.
    """

    _instance: Optional["SparkSessionManager"] = None
    _session: SparkSession | None = None
    _initialized: bool

    def __new__(cls, config: SparkConfig | None = None) -> "SparkSessionManager":
        """
        Create singleton instance.

        Args:
            config: Spark configuration. If None, uses default config.

        Returns:
            SparkSessionManager: The singleton instance
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: SparkConfig | None = None) -> None:
        """
        Initialize the manager.

        Args:
            config: Spark configuration. If None, uses default config.
        """
        # Prevent re-initialization
        if self._initialized:
            return

        from ..config import get_spark_config

        self.config = config if config is not None else get_spark_config()
        self.logger = get_logger()
        self._initialized = True

    def create_session(self) -> SparkSession:
        """
        Create and configure a new Spark session.

        Returns:
            SparkSession: Configured Spark session

        Raises:
            RuntimeError: If session creation fails
        """
        try:
            self.logger.info("Creating Spark session...")

            builder = (
                SparkSession.builder.appName(self.config.APP_NAME)
                .master(self.config.MASTER)
                .config("spark.driver.memory", self.config.DRIVER_MEMORY)
                .config("spark.executor.memory", self.config.EXECUTOR_MEMORY)
                .config("spark.executor.cores", str(self.config.EXECUTOR_CORES))
                .config("spark.sql.shuffle.partitions", str(self.config.SQL_SHUFFLE_PARTITIONS))
                .config("spark.driver.maxResultSize", self.config.MAX_RESULT_SIZE)
                .config("spark.serializer", self.config.SERIALIZER)
            )

            # Add adaptive query execution configs
            if self.config.ADAPTIVE_ENABLED:
                builder = builder.config("spark.sql.adaptive.enabled", "true")

            if self.config.ADAPTIVE_COALESCE_PARTITIONS:
                builder = builder.config("spark.sql.adaptive.coalescePartitions.enabled", "true")

            # Create or get session
            session = builder.getOrCreate()

            # Log configuration
            self.logger.info("Spark session created successfully")
            self.logger.info(f"  App Name: {self.config.APP_NAME}")
            self.logger.info(f"  Master: {self.config.MASTER}")
            self.logger.info(f"  Driver Memory: {self.config.DRIVER_MEMORY}")
            self.logger.info(f"  Executor Memory: {self.config.EXECUTOR_MEMORY}")
            self.logger.info(f"  Executor Cores: {self.config.EXECUTOR_CORES}")

            return session

        except Exception as e:
            self.logger.error(f"Failed to create Spark session: {e}")
            raise RuntimeError(f"Spark session creation failed: {e}") from e

    def get_session(self) -> SparkSession:
        """
        Get existing Spark session or create a new one.

        Returns:
            SparkSession: Active Spark session
        """
        if self._session is None:
            self._session = self.create_session()
        return self._session

    def stop_session(self) -> None:
        """Stop the active Spark session and clean up resources."""
        if self._session is not None:
            self.logger.info("Stopping Spark session...")
            try:
                self._session.stop()
                self.logger.info("Spark session stopped successfully")
            except Exception as e:
                self.logger.error(f"Error stopping Spark session: {e}")
            finally:
                self._session = None

    def __enter__(self) -> SparkSession:
        """
        Context manager entry.

        Returns:
            SparkSession: Active Spark session
        """
        return self.get_session()

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: object
    ) -> None:
        """
        Context manager exit. Stops Spark session.

        Args:
            exc_type: Exception type (if any)
            exc_val: Exception value (if any)
            exc_tb: Exception traceback (if any)
        """
        self.stop_session()

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance.

        Useful for testing or when you need to recreate the session with
        different configuration.
        """
        if cls._instance is not None:
            cls._instance.stop_session()
            cls._instance = None
            cls._session = None


def get_spark_session(config: SparkConfig | None = None) -> SparkSession:
    """
    Convenience function to get Spark session.

    Args:
        config: Optional Spark configuration

    Returns:
        SparkSession: Active Spark session
    """
    manager = SparkSessionManager(config)
    return manager.get_session()


def stop_spark_session() -> None:
    """Convenience function to stop Spark session."""
    manager = SparkSessionManager()
    manager.stop_session()
