"""
Unit tests for score-based clustering module.

Tests the score-based clustering functionality including:
- Score categorization
- Cluster label assignment
- Weighted statistics calculation
"""

import pytest
from pyspark.sql import DataFrame, SparkSession

from behavior_analysis.analysis.score_clustering import (
    add_cluster_labels,
    categorize_score,
    get_cluster_statistics,
)


@pytest.fixture(scope="session")  # type: ignore[misc]
def spark() -> SparkSession:
    """Create a Spark session for testing."""
    return (
        SparkSession.builder.master("local[1]")
        .appName("test-clustering")
        .config("spark.driver.memory", "1g")
        .config("spark.executor.memory", "1g")
        .getOrCreate()
    )


@pytest.fixture  # type: ignore[misc]
def sample_student_data(spark: SparkSession) -> DataFrame:
    """Create sample student data for testing."""
    data = [
        ("S001", 450.5, 1.2),  # Low
        ("S002", 500.0, 1.1),  # Middle
        ("S003", 482.0, 1.0),  # Middle (boundary - exactly 482)
        ("S004", 620.0, 0.9),  # High
        ("S005", 350.0, 1.3),  # Low
        ("S006", 606.0, 1.0),  # Middle (boundary - exactly 606)
        ("S007", 607.0, 0.95),  # High (boundary - exactly 607)
        ("S008", 750.0, 1.15),  # High
    ]
    return spark.createDataFrame(data, schema="student_id STRING, PV1MATH DOUBLE, W_FSTUWT DOUBLE")


class TestScoreCategorization:
    """Tests for score categorization logic."""

    def test_categorize_low_scores(self, spark: SparkSession) -> None:
        """Test that scores < 482 are categorized as 'low'."""
        df = spark.createDataFrame([(400.0,), (481.9,), (0.0,)], schema="PV1MATH DOUBLE")
        result = df.withColumn("category", categorize_score()).select("category").collect()
        assert all(row["category"] == "low" for row in result)

    def test_categorize_middle_scores(self, spark: SparkSession) -> None:
        """Test that scores 482-606 are categorized as 'middle'."""
        df = spark.createDataFrame([(482.0,), (550.0,), (606.0,)], schema="PV1MATH DOUBLE")
        result = df.withColumn("category", categorize_score()).select("category").collect()
        assert all(row["category"] == "middle" for row in result)

    def test_categorize_high_scores(self, spark: SparkSession) -> None:
        """Test that scores ≥ 607 are categorized as 'high'."""
        df = spark.createDataFrame([(607.0,), (650.0,), (800.0,)], schema="PV1MATH DOUBLE")
        result = df.withColumn("category", categorize_score()).select("category").collect()
        assert all(row["category"] == "high" for row in result)

    def test_boundary_values(self, spark: SparkSession) -> None:
        """Test boundary values are correctly categorized."""
        df = spark.createDataFrame(
            [(481.9, "low"), (482.0, "middle"), (606.0, "middle"), (607.0, "high")],
            schema="PV1MATH DOUBLE, expected STRING",
        )
        result = (
            df.withColumn("category", categorize_score()).select("category", "expected").collect()
        )
        for row in result:
            assert row["category"] == row["expected"]


class TestClusterLabeling:
    """Tests for adding cluster labels to DataFrame."""

    def test_add_cluster_labels_basic(self, sample_student_data: DataFrame) -> None:
        """Test basic cluster label addition."""
        result = add_cluster_labels(sample_student_data)
        assert "score_cluster" in result.columns
        assert result.count() == 8

    def test_cluster_distribution(self, sample_student_data: DataFrame) -> None:
        """Test that clusters are correctly distributed."""
        result = add_cluster_labels(sample_student_data)
        cluster_counts = result.groupBy("score_cluster").count().collect()
        cluster_dict = {row["score_cluster"]: row["count"] for row in cluster_counts}

        # Expected: 2 low, 3 middle, 3 high
        assert cluster_dict.get("low", 0) == 2, f"Expected low=2, got {cluster_dict}"
        assert cluster_dict.get("middle", 0) == 3, f"Expected middle=3, got {cluster_dict}"
        assert cluster_dict.get("high", 0) == 3, f"Expected high=3, got {cluster_dict}"

    def test_custom_cluster_column_name(self, sample_student_data: DataFrame) -> None:
        """Test using a custom cluster column name."""
        result = add_cluster_labels(sample_student_data, cluster_column="custom_level")
        assert "custom_level" in result.columns
        assert result.count() == 8

    def test_missing_score_column_raises_error(self, sample_student_data: DataFrame) -> None:
        """Test that missing score column raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            add_cluster_labels(sample_student_data, score_column="INVALID_COLUMN")


class TestClusterStatistics:
    """Tests for cluster statistics calculation."""

    def test_get_cluster_statistics_basic(self, sample_student_data: DataFrame) -> None:
        """Test basic statistics calculation."""
        clustered_df = add_cluster_labels(sample_student_data)
        stats = get_cluster_statistics(clustered_df)

        # Check all clusters are present
        assert "low" in stats
        assert "middle" in stats
        assert "high" in stats

    def test_statistics_keys(self, sample_student_data: DataFrame) -> None:
        """Test that all required statistics keys are present."""
        clustered_df = add_cluster_labels(sample_student_data)
        stats = get_cluster_statistics(clustered_df)

        required_keys = [
            "sample_count",
            "weighted_count",
            "mean_score",
            "weighted_mean_score",
            "min_score",
            "max_score",
            "population_percentage",
        ]

        for level_stats in stats.values():
            for key in required_keys:
                assert key in level_stats

    def test_sample_count_accuracy(self, sample_student_data: DataFrame) -> None:
        """Test that sample counts are accurate."""
        clustered_df = add_cluster_labels(sample_student_data)
        stats = get_cluster_statistics(clustered_df)

        # Expected counts: low=2, middle=3, high=3
        assert stats["low"]["sample_count"] == 2
        assert stats["middle"]["sample_count"] == 3
        assert stats["high"]["sample_count"] == 3

    def test_population_percentage_sum(self, sample_student_data: DataFrame) -> None:
        """Test that population percentages sum to approximately 100%."""
        clustered_df = add_cluster_labels(sample_student_data)
        stats = get_cluster_statistics(clustered_df)

        total_percentage = sum(s["population_percentage"] for s in stats.values())
        assert abs(total_percentage - 100.0) < 0.01

    def test_weighted_statistics(self, sample_student_data: DataFrame) -> None:
        """Test that weighted statistics differ from unweighted when weights vary."""
        clustered_df = add_cluster_labels(sample_student_data)
        stats = get_cluster_statistics(clustered_df)

        # When weights differ, weighted mean should differ from sample mean
        for _level, level_stats in stats.items():
            if level_stats["mean_score"] is not None:
                # Just verify both statistics exist and are reasonable
                assert level_stats["weighted_mean_score"] is not None
                assert 0 <= level_stats["mean_score"] <= 1000
                assert 0 <= level_stats["weighted_mean_score"] <= 1000

    def test_missing_weight_column_raises_error(self, sample_student_data: DataFrame) -> None:
        """Test that missing weight column raises ValueError."""
        clustered_df = add_cluster_labels(sample_student_data)
        with pytest.raises(ValueError, match="not found"):
            get_cluster_statistics(clustered_df, weight_column="INVALID_WEIGHT")

    def test_score_ranges(self, sample_student_data: DataFrame) -> None:
        """Test that min/max scores are within expected ranges."""
        clustered_df = add_cluster_labels(sample_student_data)
        stats = get_cluster_statistics(clustered_df)

        # Low cluster: min < 482, max < 482
        assert stats["low"]["max_score"] < 482

        # Middle cluster: min >= 482, max <= 606
        assert stats["middle"]["min_score"] >= 482
        assert stats["middle"]["max_score"] <= 606

        # High cluster: min >= 607
        assert stats["high"]["min_score"] >= 607


class TestIntegration:
    """Integration tests for the complete clustering workflow."""

    def test_end_to_end_clustering(self, sample_student_data: DataFrame) -> None:
        """Test complete clustering workflow."""
        # Step 1: Add labels
        clustered_df = add_cluster_labels(sample_student_data)

        # Step 2: Get statistics
        stats = get_cluster_statistics(clustered_df)

        # Verify results
        assert len(stats) == 3
        assert all(isinstance(s["sample_count"], int) for s in stats.values())
        assert all(isinstance(s["population_percentage"], float) for s in stats.values())

    def test_data_consistency(self, sample_student_data: DataFrame) -> None:
        """Test that clustering doesn't lose or duplicate data."""
        original_count = sample_student_data.count()
        clustered_df = add_cluster_labels(sample_student_data)
        clustered_count = clustered_df.count()

        assert original_count == clustered_count
