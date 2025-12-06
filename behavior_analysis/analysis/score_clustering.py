"""
Score-based clustering analysis module.

Implements score-based clustering analysis for PISA student math performance.
Students are grouped into three levels based on PV1MATH scores:
- Low: < 482
- Middle: 482-606
- High: ≥ 607

Uses sampling weights (W_FSTUWT) for global population statistics.
"""

from typing import Any

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f

from ..utils.logger import get_logger

# PISA Math Score Thresholds (Official Levels)
SCORE_THRESHOLDS = {
    "low": (0, 482),  # < 482
    "middle": (482, 607),  # 482-606
    "high": (607, float("inf")),  # ≥ 607
}


def categorize_score(score_column: str = "PV1MATH") -> Column:
    """
    Create a Spark SQL expression to categorize scores into levels.

    Args:
        score_column: Name of the score column (default: PV1MATH)

    Returns:
        Spark Column expression for score categorization
    """
    return (
        f.when(f.col(score_column) < 482, "low")
        .when((f.col(score_column) >= 482) & (f.col(score_column) < 607), "middle")
        .otherwise("high")
    )


def add_cluster_labels(
    df: DataFrame, score_column: str = "PV1MATH", cluster_column: str = "score_cluster"
) -> DataFrame:
    """
    Add cluster labels to the DataFrame based on score thresholds.

    Args:
        df: Input Spark DataFrame
        score_column: Name of the score column to cluster on
        cluster_column: Name of the new cluster column (default: score_cluster)

    Returns:
        DataFrame with added cluster column
    """
    logger = get_logger()

    # Validate columns exist
    if score_column not in df.columns:
        raise ValueError(f"Score column '{score_column}' not found in dataset")

    logger.info("Adding cluster labels based on %s", score_column)
    logger.info("  Low: < 482")
    logger.info("  Middle: 482-606")
    logger.info("  High: ≥ 607")

    return df.withColumn(cluster_column, categorize_score(score_column))


def get_cluster_statistics(
    df: DataFrame,
    score_column: str = "PV1MATH",
    weight_column: str = "W_FSTUWT",
    cluster_column: str = "score_cluster",
) -> dict[str, Any]:
    """
    Calculate weighted statistics for each score cluster.

    Statistics include:
    - Count: Number of students in cluster
    - Weighted Count: Sum of weights (representing global population)
    - Mean Score: Average score in cluster
    - Weighted Mean Score: Population-weighted average
    - Min/Max Score: Score range
    - Percentage: Percentage of global population

    Args:
        df: Input Spark DataFrame (should include PV1MATH and W_FSTUWT columns)
        score_column: Name of the score column (default: PV1MATH)
        weight_column: Name of the sampling weight column (default: W_FSTUWT)
        cluster_column: Name of the cluster column (default: score_cluster)

    Returns:
        Dict[str, Any]: Statistics grouped by cluster level
    """
    logger = get_logger()

    # Validate columns exist
    for col in [score_column, weight_column, cluster_column]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataset")

    logger.info("Computing weighted statistics for clusters")
    logger.info("  Score column: %s", score_column)
    logger.info("  Weight column: %s", weight_column)

    # Calculate cluster statistics
    cluster_stats = (
        df.groupBy(cluster_column)
        .agg(
            f.count(f.col(score_column)).alias("count"),
            f.sum(f.col(weight_column)).alias("weighted_count"),
            f.mean(f.col(score_column)).alias("mean_score"),
            (f.sum(f.col(score_column) * f.col(weight_column)) / f.sum(f.col(weight_column))).alias(
                "weighted_mean_score"
            ),
            f.min(f.col(score_column)).alias("min_score"),
            f.max(f.col(score_column)).alias("max_score"),
        )
        .collect()
    )

    # Calculate total weighted count for percentage calculation
    total_weighted = df.select(f.sum(f.col(weight_column)).alias("total")).collect()[0]["total"]

    # Format results
    result = {}
    for row in cluster_stats:
        cluster_level = row[cluster_column]
        weighted_count = row["weighted_count"]

        result[cluster_level] = {
            "sample_count": row["count"],
            "weighted_count": float(weighted_count) if weighted_count else 0,
            "mean_score": (float(row["mean_score"]) if row["mean_score"] is not None else None),
            "weighted_mean_score": (
                float(row["weighted_mean_score"])
                if row["weighted_mean_score"] is not None
                else None
            ),
            "min_score": (float(row["min_score"]) if row["min_score"] is not None else None),
            "max_score": (float(row["max_score"]) if row["max_score"] is not None else None),
            "population_percentage": (
                (float(weighted_count) / total_weighted * 100) if total_weighted else 0
            ),
        }

    logger.info("Cluster statistics computed successfully")
    return result


def print_clustering_report(statistics: dict[str, Any], verbose: bool = True) -> None:
    """
    Print a formatted clustering analysis report.

    Args:
        statistics: Cluster statistics dictionary from get_cluster_statistics()
        verbose: If True, print detailed statistics (default: True)
    """
    print("\n" + "=" * 80, flush=True)
    print("SCORE-BASED CLUSTERING ANALYSIS REPORT", flush=True)
    print("=" * 80, flush=True)

    # Define cluster order and names
    cluster_order = ["low", "middle", "high"]
    cluster_names = {
        "low": "Low (< 482)",
        "middle": "Middle (482-606)",
        "high": "High (≥ 607)",
    }

    for level in cluster_order:
        if level not in statistics:
            continue

        stats = statistics[level]
        print(f"\n{cluster_names[level]}", flush=True)
        print("-" * 80, flush=True)
        print(f"  Sample Size:           {stats['sample_count']:,} students", flush=True)
        print(
            f"  Weighted Population:   {stats['weighted_count']:,.0f} students",
            flush=True,
        )
        print(
            f"  Population Share:      {stats['population_percentage']:.2f}%",
            flush=True,
        )

        if verbose:
            mean_msg = (
                f"  Mean Score (sample):   {stats['mean_score']:.2f}"
                if stats["mean_score"] is not None
                else "  Mean Score (sample):   N/A"
            )
            print(mean_msg, flush=True)
            weighted_msg = (
                f"  Mean Score (weighted): {stats['weighted_mean_score']:.2f}"
                if stats["weighted_mean_score"] is not None
                else "  Mean Score (weighted): N/A"
            )
            print(weighted_msg, flush=True)
            range_msg = (
                f"  Score Range:           {stats['min_score']:.0f} - {stats['max_score']:.0f}"
                if stats["min_score"] is not None
                else "  Score Range:           N/A"
            )
            print(range_msg, flush=True)

    print("\n" + "=" * 80, flush=True)


def export_cluster_data(
    df: DataFrame,
    output_path: str,
) -> None:
    """
    Export clustered data to Parquet format.

    Args:
        df: DataFrame with cluster labels
        output_path: Output path for Parquet files
    """
    logger = get_logger()

    logger.info("Exporting clustered data to: %s", output_path)
    df.write.mode("overwrite").parquet(output_path)
    logger.info("Clustered data exported successfully")
