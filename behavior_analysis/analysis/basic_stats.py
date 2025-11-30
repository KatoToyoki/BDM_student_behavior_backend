"""
Basic statistical analysis module.

Provides functions for computing statistical measures on Spark DataFrames.
"""

from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as f

from ..utils.logger import get_logger


def calculate_column_mean(df: DataFrame, column_name: str) -> float | None:
    """
    Calculate the mean (average) of a column.

    Args:
        df: Spark DataFrame
        column_name: Name of column to calculate mean for

    Returns:
        Optional[float]: Mean value, or None if column not found or empty

    Raises:
        ValueError: If column doesn't exist in DataFrame
    """
    logger = get_logger()

    # Validate column exists
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in dataset")

    logger.info(f"Calculating mean for column: {column_name}")

    # Calculate mean (automatically excludes NULL values)
    result = df.select(f.mean(f.col(column_name)).alias("mean")).collect()

    if result and result[0]["mean"] is not None:
        mean_value = float(result[0]["mean"])
        logger.info(f"Mean of {column_name}: {mean_value}")
        return mean_value

    logger.warning(f"No valid values found for column: {column_name}")
    return None


def get_column_statistics(df: DataFrame, column_name: str) -> dict[str, Any]:
    """
    Get comprehensive statistics for a column.

    Computes:
    - count: Number of non-null values
    - mean: Average value
    - stddev: Standard deviation
    - min: Minimum value
    - max: Maximum value
    - null_count: Number of null values

    Args:
        df: Spark DataFrame
        column_name: Name of column to analyze

    Returns:
        Dict[str, Any]: Dictionary containing statistics

    Raises:
        ValueError: If column doesn't exist in DataFrame
    """
    logger = get_logger()

    # Validate column exists
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in dataset")

    logger.info(f"Computing statistics for column: {column_name}")

    # Calculate statistics
    stats = df.select(
        f.count(f.col(column_name)).alias("count"),
        f.mean(f.col(column_name)).alias("mean"),
        f.stddev(f.col(column_name)).alias("stddev"),
        f.min(f.col(column_name)).alias("min"),
        f.max(f.col(column_name)).alias("max"),
    ).collect()[0]

    # Count nulls
    null_count = df.select(
        f.sum(f.when(f.col(column_name).isNull(), 1).otherwise(0)).alias("null_count")
    ).collect()[0]["null_count"]

    total_rows = df.count()

    result = {
        "column": column_name,
        "count": stats["count"],
        "mean": float(stats["mean"]) if stats["mean"] is not None else None,
        "stddev": float(stats["stddev"]) if stats["stddev"] is not None else None,
        "min": float(stats["min"]) if stats["min"] is not None else None,
        "max": float(stats["max"]) if stats["max"] is not None else None,
        "null_count": null_count,
        "total_rows": total_rows,
        "null_percentage": (null_count / total_rows * 100) if total_rows > 0 else 0,
    }

    logger.info(f"Statistics computed for {column_name}")
    return result


def describe_dataset(df: DataFrame, columns: list[str] | None = None) -> dict[str, Any]:
    """
    Generate a summary description of the dataset.

    Args:
        df: Spark DataFrame
        columns: List of column names to describe. If None, uses numeric columns.

    Returns:
        Dict[str, Any]: Dataset description including row count, column info, etc.
    """
    logger = get_logger()

    logger.info("Generating dataset description...")

    # Get basic info
    total_rows = df.count()
    total_columns = len(df.columns)

    # If no columns specified, use numeric columns
    if columns is None:
        numeric_types = ["int", "bigint", "float", "double", "decimal"]
        columns = [
            field.name
            for field in df.schema.fields
            if any(t in str(field.dataType).lower() for t in numeric_types)
        ]

    # Get Spark's describe() output
    describe_df = df.describe(columns) if columns else df.describe()

    # Convert to dict format
    summary_rows = describe_df.collect()
    summary_dict = {}

    for row in summary_rows:
        stat_name = row["summary"]
        summary_dict[stat_name] = {col: row[col] for col in columns}

    result = {
        "total_rows": total_rows,
        "total_columns": total_columns,
        "analyzed_columns": columns,
        "summary": summary_dict,
    }

    logger.info(f"Dataset description completed: {total_rows:,} rows, {total_columns} columns")
    return result


def get_missing_values_report(df: DataFrame) -> dict[str, dict[str, Any]]:
    """
    Generate a report of missing values for all columns.

    Args:
        df: Spark DataFrame

    Returns:
        Dict[str, Dict[str, Any]]: Report with column names as keys
    """
    logger = get_logger()

    logger.info("Generating missing values report...")

    total_rows = df.count()
    report = {}

    for column in df.columns:
        null_count = df.select(
            f.sum(f.when(f.col(column).isNull(), 1).otherwise(0)).alias("null_count")
        ).collect()[0]["null_count"]

        null_percentage = (null_count / total_rows * 100) if total_rows > 0 else 0

        report[column] = {
            "null_count": null_count,
            "non_null_count": total_rows - null_count,
            "null_percentage": null_percentage,
        }

    logger.info("Missing values report completed")
    return report


def calculate_correlation(
    df: DataFrame, col1: str, col2: str, method: str = "pearson"
) -> float | None:
    """
    Calculate correlation between two columns.

    Args:
        df: Spark DataFrame
        col1: First column name
        col2: Second column name
        method: Correlation method ("pearson" or "spearman")

    Returns:
        Optional[float]: Correlation coefficient

    Raises:
        ValueError: If columns don't exist or method is invalid
    """
    logger = get_logger()

    # Validate columns
    if col1 not in df.columns:
        raise ValueError(f"Column '{col1}' not found in dataset")
    if col2 not in df.columns:
        raise ValueError(f"Column '{col2}' not found in dataset")

    # Validate method
    if method not in ["pearson", "spearman"]:
        raise ValueError(f"Invalid method: {method}. Use 'pearson' or 'spearman'")

    logger.info(f"Calculating {method} correlation between {col1} and {col2}")

    # Calculate correlation
    correlation = df.stat.corr(col1, col2, method=method)

    logger.info(f"Correlation: {correlation}")
    return float(correlation) if correlation is not None else None


def print_statistics_report(stats: dict[str, Any]) -> None:
    """
    Print a formatted statistics report.

    Args:
        stats: Statistics dictionary from get_column_statistics()
    """
    print("\n" + "=" * 60)
    print(f"Statistics Report: {stats['column']}")
    print("=" * 60)
    print(f"Total Rows:        {stats['total_rows']:,}")
    print(f"Non-Null Count:    {stats['count']:,}")
    print(f"Null Count:        {stats['null_count']:,} ({stats['null_percentage']:.2f}%)")
    print("-" * 60)

    if stats["mean"] is not None:
        print(f"Mean:              {stats['mean']:.4f}")
    if stats["stddev"] is not None:
        print(f"Std Dev:           {stats['stddev']:.4f}")
    if stats["min"] is not None:
        print(f"Min:               {stats['min']:.4f}")
    if stats["max"] is not None:
        print(f"Max:               {stats['max']:.4f}")

    print("=" * 60 + "\n")
