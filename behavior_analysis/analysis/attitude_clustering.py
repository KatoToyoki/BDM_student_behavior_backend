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

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import DataFrame
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


def prepare_attitude_data(
    df: DataFrame, weight_column: str = "W_FSTUWT"
) -> tuple[DataFrame, dict[str, Any]]:
    """
    Prepare attitude data by handling missing values and outliers.

    Args:
        df: Input Spark DataFrame with attitude columns
        weight_column: Name of the sampling weight column

    Returns:
        Tuple of (cleaned DataFrame, missing value statistics dict)
    """
    logger = get_logger()

    # Handle missing values - drop rows with any missing attitude data
    attitude_cols = list(ATTITUDE_DIMENSIONS.keys())

    # Calculate missing value statistics for each variable
    logger.info("Calculating missing value statistics...")
    missing_stats = {}
    total_count = df.count()

    for col_name in attitude_cols:
        null_count = df.filter(f.col(col_name).isNull()).count()
        missing_rate = (null_count / total_count * 100) if total_count > 0 else 0
        missing_stats[col_name] = {
            "missing_count": null_count,
            "missing_rate": missing_rate,
            "valid_count": total_count - null_count,
        }
        logger.info(f"  {col_name}: {null_count} missing ({missing_rate:.2f}%)")

    # Build condition to drop rows where any attitude column is null
    from pyspark.sql import Column

    condition: Column | None = None
    for col in attitude_cols:
        col_condition = f.col(col).isNotNull()
        condition = col_condition if condition is None else condition & col_condition

    # Calculate weighted population before and after cleaning
    total_weighted = df.select(f.sum(f.col(weight_column)).alias("total")).collect()[0]["total"]
    total_weighted = float(total_weighted) if total_weighted else 0

    # Filter with type assertion since we know condition is not None after the loop
    assert condition is not None
    df_clean = df.filter(condition)
    clean_count = df_clean.count()
    clean_weighted = df_clean.select(f.sum(f.col(weight_column)).alias("total")).collect()[0][
        "total"
    ]
    clean_weighted = float(clean_weighted) if clean_weighted else 0

    # Calculate loss statistics
    removed_count = total_count - clean_count
    removed_weighted = total_weighted - clean_weighted
    loss_rate = (removed_count / total_count * 100) if total_count > 0 else 0
    weighted_loss_rate = (removed_weighted / total_weighted * 100) if total_weighted > 0 else 0

    logger.info(f"Original rows: {total_count:,}")
    logger.info(f"Rows after removing missing values: {clean_count:,}")
    logger.info(f"Rows removed: {removed_count:,} ({loss_rate:.2f}%)")
    logger.info(f"Original weighted population: {total_weighted:,.0f}")
    logger.info(f"Cleaned weighted population: {clean_weighted:,.0f}")
    logger.info(f"Weighted population lost: {removed_weighted:,.0f} ({weighted_loss_rate:.2f}%)")

    # Compile statistics
    statistics = {
        "variable_missing_rates": missing_stats,
        "sample_loss": {
            "original_count": total_count,
            "cleaned_count": clean_count,
            "removed_count": removed_count,
            "loss_rate": loss_rate,
        },
        "weighted_loss": {
            "original_weighted": total_weighted,
            "cleaned_weighted": clean_weighted,
            "removed_weighted": removed_weighted,
            "weighted_loss_rate": weighted_loss_rate,
        },
    }

    return df_clean, statistics


def create_attitude_features(df: DataFrame) -> DataFrame:
    """
    Create feature vector for K-means clustering.

    Inverts ST062Q01TA (School Discipline) so higher values = better behavior.
    Uses VectorAssembler to combine attitude dimensions into a feature vector.

    Args:
        df: Input Spark DataFrame with attitude columns

    Returns:
        DataFrame with 'attitude_features' column (Vector type)
    """
    logger = get_logger()

    logger.info("Creating attitude feature vector")
    logger.info(f"Dimensions: {', '.join(ATTITUDE_DIMENSIONS.values())}")

    # Invert ST062Q01TA (School Discipline) since it's a negative indicator
    # Original: 1=Never skip, 4=Skip often → Inverted: Higher=Better
    max_st062_result = df.agg(f.max("ST062Q01TA")).collect()[0][0]
    if max_st062_result is None:
        raise ValueError("ST062Q01TA column contains only null values")
    max_st062 = float(max_st062_result)
    df_inverted = df.withColumn("ST062Q01TA_inverted", max_st062 + 1 - f.col("ST062Q01TA"))

    logger.info(
        f"Inverted ST062Q01TA: max value = {max_st062}, using formula: {max_st062 + 1} - ST062Q01TA"
    )

    # Prepare column list with inverted ST062Q01TA
    feature_cols = [
        "ST296Q01JA",
        "ST062Q01TA_inverted",  # Use inverted version
        "MATHMOT",
        "PERSEVAGR",
    ]

    # Assemble features into vector
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="attitude_features_raw",
        handleInvalid="skip",  # Skip rows with null values
    )

    df_with_features = assembler.transform(df_inverted)

    # Standardize features using Z-score normalization
    scaler = StandardScaler(
        inputCol="attitude_features_raw",
        outputCol="attitude_features",
        withMean=True,
        withStd=True,
    )

    df_scaled = scaler.fit(df_with_features).transform(df_with_features)

    logger.info("Features standardized using Z-score normalization (equal weighting)")
    logger.info("All dimensions now positively oriented (higher = better)")

    return df_scaled


def perform_attitude_clustering(
    df: DataFrame, num_clusters: int = 3
) -> tuple[DataFrame, dict[int, str]]:
    """
    Perform K-means clustering on standardized attitude features.

    Args:
        df: Input Spark DataFrame with 'attitude_features' column
        num_clusters: Number of clusters (default: 3)

    Returns:
        Tuple of (DataFrame with 'attitude_cluster' column, label mapping dict)
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

    logger.info("K-means clustering completed")

    # Get cluster centers and auto-assign labels
    centers = model.clusterCenters()
    # Convert numpy arrays to list of lists for type compatibility
    centers_list = [center.tolist() for center in centers]
    label_mapping = _assign_cluster_labels(centers_list)

    logger.info("Cluster centers and assigned labels:")
    for cluster_id, center in enumerate(centers):
        logger.info(f"  Cluster {cluster_id} -> {label_mapping[cluster_id]}")
        logger.info(f"    Features: {center}")

    return df_clustered, label_mapping


def _assign_cluster_labels(cluster_centers: list[list[float]]) -> dict[int, str]:
    """
    Automatically assign attitude labels based on cluster center characteristics.

    Strategy: Calculate an "engagement score" for each cluster based on the
    standardized feature values (Z-scores). Since ST062Q01TA has been inverted
    during feature creation, all features are now positively oriented.

    Args:
        cluster_centers: List of cluster center vectors from K-means

    Returns:
        Dictionary mapping cluster ID to attitude label
    """
    logger = get_logger()

    # Calculate engagement score for each cluster (sum of standardized features)
    # All features are now positive indicators (higher = better):
    # Features order: ST296Q01JA, ST062Q01TA_inverted, MATHMOT, PERSEVAGR
    engagement_scores = []
    for i, center in enumerate(cluster_centers):
        # Simple sum since all dimensions are positive
        score = sum(center)
        engagement_scores.append((i, score))
        logger.info(f"Cluster {i} engagement score: {score:.3f}")

    # Sort by engagement score (descending)
    sorted_clusters = sorted(engagement_scores, key=lambda x: x[1], reverse=True)

    # Assign labels based on ranking
    label_mapping = {}
    num_clusters = len(cluster_centers)

    # Define labels for different cluster counts
    if num_clusters == 2:
        labels = ["Proactive Learners", "Disengaged Learners"]
    elif num_clusters == 3:
        labels = ["Proactive Learners", "Average Learners", "Disengaged Learners"]
    else:
        # For k > 3, generate labels like "Cluster 1 (High)", "Cluster 2 (Medium-High)", etc.
        labels = [f"Cluster {i + 1} (Engagement Rank {i + 1})" for i in range(num_clusters)]

    for rank, (cluster_id, score) in enumerate(sorted_clusters):
        label_mapping[cluster_id] = labels[rank]
        logger.info(f"  Cluster {cluster_id} (score: {score:.3f}) -> {labels[rank]}")

    return label_mapping


def add_attitude_labels(
    df: DataFrame,
    label_mapping: dict[int, str],
    cluster_column: str = "attitude_cluster",
) -> DataFrame:
    """
    Map cluster IDs to human-readable attitude labels.

    Args:
        df: DataFrame with cluster assignments
        label_mapping: Dictionary mapping cluster IDs to labels (from perform_attitude_clustering)
        cluster_column: Name of the cluster column

    Returns:
        DataFrame with 'attitude_label' column
    """
    logger = get_logger()

    logger.info("Mapping clusters to attitude labels:")
    for cluster_id, label in label_mapping.items():
        logger.info(f"  Cluster {cluster_id}: {label}")

    # Create mapping expression
    first_id = list(label_mapping.keys())[0]
    cluster_case = f.when(f.col(cluster_column) == first_id, label_mapping[first_id])
    for cluster_id, label in list(label_mapping.items())[1:]:
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
    total_weighted = df.select(f.sum(f.col(weight_column)).alias("total")).collect()[0]["total"]

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
        print(f"  Sample Size:           {stats['sample_count']:,} students", flush=True)
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
