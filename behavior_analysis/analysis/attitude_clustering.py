"""
Attitude-based clustering analysis module.

Implements attitude-based clustering for PISA student behavioral engagement.
Students are grouped into three attitude clusters using K-means clustering
on four behavioral engagement dimensions:
- Learning Time (ST296Q01JA): hours per week
- School Discipline (ST062Q01TA): skipping school days (negative indicator)
- Math Motivation (MATHMOT): motivation index
- Perseverance (ST307): persistence trait

Uses Z-score standardization without weighting to ensure equal contribution
from all four dimensions.
"""

from typing import Any

import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f

from ..utils.logger import get_logger

# Attitude dimensions for clustering
ATTITUDE_DIMENSIONS = {
    "ST296Q01JA": "Learning Time",
    "ST062Q01TA": "School Discipline (inverted)",
    "MATHMOT": "Math Motivation",
    "PERSEVAGR": "Perseverance",
}

# Cluster labels for attitude groups
ATTITUDE_CLUSTER_LABELS = {
    0: "Proactive Learners",
    1: "Average Learners",
    2: "Disengaged Learners",
}


def validate_attitude_columns(df: DataFrame) -> None:
    """
    Validate that all required attitude columns exist in the dataset.

    Args:
        df: Input Spark DataFrame

    Raises:
        ValueError: If any required column is missing
    """
    required_cols = list(ATTITUDE_DIMENSIONS.keys())
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required attitude columns: {missing_cols}")


def prepare_attitude_data(df: DataFrame) -> DataFrame:
    """
    Prepare attitude data by handling missing values and outliers.

    Args:
        df: Input Spark DataFrame with attitude columns

    Returns:
        DataFrame with cleaned attitude data
    """
    logger = get_logger()

    # Handle missing values - drop rows with any missing attitude data
    attitude_cols = list(ATTITUDE_DIMENSIONS.keys())

    # Build condition to drop rows where any attitude column is null
    condition = None
    for col in attitude_cols:
        col_condition = f.col(col).isNotNull()
        condition = col_condition if condition is None else condition & col_condition

    original_count = df.count()
    df_clean = df.filter(condition)
    clean_count = df_clean.count()

    logger.info(f"Original rows: {original_count}")
    logger.info(f"Rows after removing missing values: {clean_count}")
    logger.info(f"Rows removed: {original_count - clean_count}")

    return df_clean


def create_attitude_features(df: DataFrame) -> DataFrame:
    """
    Create feature vector for K-means clustering.

    Uses VectorAssembler to combine attitude dimensions into a feature vector.

    Args:
        df: Input Spark DataFrame with attitude columns

    Returns:
        DataFrame with 'attitude_features' column (Vector type)
    """
    logger = get_logger()

    attitude_cols = list(ATTITUDE_DIMENSIONS.keys())

    logger.info("Creating attitude feature vector")
    logger.info(f"Dimensions: {', '.join(ATTITUDE_DIMENSIONS.values())}")

    # Assemble features into vector
    assembler = VectorAssembler(
        inputCols=attitude_cols,
        outputCol="attitude_features_raw",
        handleInvalid="skip",  # Skip rows with null values
    )

    df_with_features = assembler.transform(df)

    # Standardize features using Z-score normalization
    scaler = StandardScaler(
        inputCol="attitude_features_raw",
        outputCol="attitude_features",
        withMean=True,
        withStd=True,
    )

    df_scaled = scaler.fit(df_with_features).transform(df_with_features)

    logger.info("Features standardized using Z-score normalization (equal weighting)")

    return df_scaled


def perform_attitude_clustering(df: DataFrame, num_clusters: int = 3) -> DataFrame:
    """
    Perform K-means clustering on standardized attitude features.

    Args:
        df: Input Spark DataFrame with 'attitude_features' column
        num_clusters: Number of clusters (default: 3)

    Returns:
        DataFrame with 'attitude_cluster' column (0, 1, or 2)
    """
    logger = get_logger()

    logger.info(f"Performing K-means clustering with k={num_clusters}")

    # K-means clustering
    kmeans = KMeans(
        k=num_clusters,
        seed=42,  # Fixed seed for reproducibility
        maxIter=20,
        initMode="k-means||",
        featuresCol="attitude_features",  # Use the scaled features column
        predictionCol="attitude_cluster",  # Output cluster assignment column
    )

    model = kmeans.fit(df)
    df_clustered = model.transform(df)

    logger.info(f"K-means clustering completed")
    logger.info(f"Cluster centers: {model.clusterCenters()}")

    return df_clustered


def add_attitude_labels(
    df: DataFrame, cluster_column: str = "attitude_cluster"
) -> DataFrame:
    """
    Map cluster IDs to human-readable attitude labels.

    Args:
        df: DataFrame with cluster assignments
        cluster_column: Name of the cluster column

    Returns:
        DataFrame with 'attitude_label' column
    """
    logger = get_logger()

    logger.info("Mapping clusters to attitude labels:")
    for cluster_id, label in ATTITUDE_CLUSTER_LABELS.items():
        logger.info(f"  Cluster {cluster_id}: {label}")

    # Create mapping expression
    cluster_case = f.when(f.col(cluster_column) == 0, ATTITUDE_CLUSTER_LABELS[0])
    for cluster_id, label in list(ATTITUDE_CLUSTER_LABELS.items())[1:]:
        cluster_case = cluster_case.when(f.col(cluster_column) == cluster_id, label)

    df_labeled = df.withColumn("attitude_label", cluster_case)

    return df_labeled


def get_attitude_statistics(
    df: DataFrame,
    weight_column: str = "W_FSTUWT",
    cluster_column: str = "attitude_cluster",
) -> dict[str, Any]:
    """
    Calculate weighted statistics for each attitude cluster.

    Args:
        df: Input DataFrame with clusters and weights
        weight_column: Name of the sampling weight column
        cluster_column: Name of the cluster column

    Returns:
        Dictionary with statistics for each cluster
    """
    logger = get_logger()

    # Validate columns exist
    for col in [weight_column, cluster_column, "attitude_label"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataset")

    logger.info("Computing weighted statistics for attitude clusters")
    logger.info(f"  Weight column: {weight_column}")

    # Calculate cluster statistics
    cluster_stats = (
        df.groupBy(cluster_column, "attitude_label")
        .agg(
            f.count("*").alias("count"),
            f.sum(f.col(weight_column)).alias("weighted_count"),
        )
        .collect()
    )

    # Calculate total weighted count
    total_weighted = df.select(f.sum(f.col(weight_column)).alias("total")).collect()[0][
        "total"
    ]

    # Format results
    result = {}
    for row in cluster_stats:
        cluster_id = row[cluster_column]
        cluster_label = row["attitude_label"]
        weighted_count = row["weighted_count"]

        result[cluster_label] = {
            "cluster_id": cluster_id,
            "sample_count": row["count"],
            "weighted_count": float(weighted_count) if weighted_count else 0,
            "population_percentage": (
                (float(weighted_count) / total_weighted * 100) if total_weighted else 0
            ),
        }

    logger.info("Attitude cluster statistics computed successfully")
    return result


def print_attitude_report(statistics: dict[str, Any], verbose: bool = True) -> None:
    """
    Print a formatted attitude clustering report.

    Args:
        statistics: Cluster statistics dictionary
        verbose: If True, print detailed statistics
    """
    print("\n" + "=" * 80, flush=True)
    print("ATTITUDE-BASED CLUSTERING ANALYSIS REPORT", flush=True)
    print("=" * 80, flush=True)

    cluster_order = ["Proactive Learners", "Average Learners", "Disengaged Learners"]

    for label in cluster_order:
        if label not in statistics:
            continue

        stats = statistics[label]
        print(f"\n{label}", flush=True)
        print("-" * 80, flush=True)
        print(
            f"  Sample Size:           {stats['sample_count']:,} students", flush=True
        )
        print(
            f"  Weighted Population:   {stats['weighted_count']:,.0f} students",
            flush=True,
        )
        print(
            f"  Population Share:      {stats['population_percentage']:.2f}%",
            flush=True,
        )

    print("\n" + "=" * 80, flush=True)


def export_attitude_data(
    df: DataFrame,
    output_path: str,
) -> None:
    """
    Export attitude-clustered data to Parquet format.

    Args:
        df: DataFrame with attitude clusters
        output_path: Output path for Parquet files
    """
    logger = get_logger()

    logger.info(f"Exporting attitude-clustered data to: {output_path}")
    df.write.mode("overwrite").parquet(output_path)
    logger.info("Attitude-clustered data exported successfully")
