"""
Score-Attitude Cross-Dimensional Analysis Module.

Analyzes the relationship between attitude-based clusters and score-based clusters
using cross-tabulation, chi-square tests, and multiple visualization techniques.
"""

from typing import Any

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as f
from scipy.stats import chi2_contingency

from ..utils.logger import get_logger

# Cluster labels mapping
SCORE_LABELS = {
    "low": "Low (< 482)",
    "middle": "Middle (482-606)",
    "high": "High (≥ 607)",
}

ATTITUDE_LABELS = {
    0: "Proactive Learners",
    1: "Average Learners",
    2: "Disengaged Learners",
}


def create_cross_tabulation(
    df: DataFrame,
    score_column: str = "score_cluster",
    attitude_column: str = "attitude_cluster",
    weight_column: str = "W_FSTUWT",
    score_label_column: str = "score_label",
    attitude_label_column: str = "attitude_label",
) -> pd.DataFrame:
    """
    Create a cross-tabulation table with weighted values.

    Args:
        df: Input DataFrame with score and attitude clusters
        score_column: Name of the score cluster column
        attitude_column: Name of the attitude cluster column
        weight_column: Name of the sampling weight column
        score_label_column: Name of the score label column
        attitude_label_column: Name of the attitude label column

    Returns:
        Pandas DataFrame with cross-tabulation (weighted counts)
    """
    logger = get_logger()

    logger.info("Creating cross-tabulation table with weights")

    # Aggregate weighted counts by score and attitude cluster
    cross_tab_data = (
        df.groupBy(score_label_column, attitude_label_column)
        .agg(f.sum(f.col(weight_column)).alias("weighted_count"))
        .collect()
    )

    # Convert to pandas and pivot
    pd_data = []
    for row in cross_tab_data:
        pd_data.append(
            {
                "Score Cluster": row[score_label_column],
                "Attitude Cluster": row[attitude_label_column],
                "Weighted Count": float(row["weighted_count"]),
            }
        )

    df_cross = pd.DataFrame(pd_data)

    # Pivot table
    cross_tab = df_cross.pivot_table(
        index="Score Cluster",
        columns="Attitude Cluster",
        values="Weighted Count",
        fill_value=0,
    )

    logger.info("Cross-tabulation table created successfully")
    logger.info(f"Shape: {cross_tab.shape}")

    return cross_tab


def perform_chi_square_test(
    cross_tab: pd.DataFrame,
) -> dict[str, Any]:
    """
    Perform chi-square test on cross-tabulation table.

    Args:
        cross_tab: Cross-tabulation table from create_cross_tabulation()

    Returns:
        Dictionary with chi-square test results
    """
    logger = get_logger()

    logger.info("Performing chi-square test")

    # Perform chi-square test
    chi2, p_value, dof, expected = chi2_contingency(cross_tab)

    logger.info(f"Chi-square statistic: {chi2:.4f}")
    logger.info(f"P-value: {p_value:.6f}")
    logger.info(f"Degrees of freedom: {dof}")

    # Determine significance
    is_significant = p_value < 0.05

    results = {
        "chi2_statistic": chi2,
        "p_value": p_value,
        "degrees_of_freedom": dof,
        "is_significant": is_significant,
        "expected_frequencies": expected,
        "interpretation": (
            "Significant association found"
            if is_significant
            else "No significant association"
        ),
    }

    return results


def export_cross_tabulation(
    cross_tab: pd.DataFrame,
    output_path: str = "visualizations/attitude_score_crosstab.csv",
) -> str:
    """
    Export cross-tabulation table to CSV.

    Args:
        cross_tab: Cross-tabulation table
        output_path: Output file path

    Returns:
        Path to saved CSV file
    """
    logger = get_logger()

    from pathlib import Path

    full_path = Path(output_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)

    cross_tab.to_csv(full_path)

    logger.info(f"Cross-tabulation exported to {output_path}")
    return str(full_path)


def print_chi_square_report(results: dict[str, Any]) -> None:
    """
    Print formatted chi-square test report.

    Args:
        results: Chi-square test results dictionary
    """
    print("\n" + "=" * 80, flush=True)
    print("CHI-SQUARE TEST RESULTS", flush=True)
    print("=" * 80, flush=True)
    print(f"\nChi-Square Statistic: {results['chi2_statistic']:.4f}", flush=True)
    print(f"P-Value: {results['p_value']:.6f}", flush=True)
    print(f"Degrees of Freedom: {results['degrees_of_freedom']}", flush=True)
    print(
        f"\nSignificance Level (α=0.05): {'Significant' if results['is_significant'] else 'Not Significant'}",
        flush=True,
    )
    print(f"Interpretation: {results['interpretation']}", flush=True)
    print("=" * 80 + "\n", flush=True)


def print_cross_tabulation(cross_tab: pd.DataFrame) -> None:
    """
    Print formatted cross-tabulation table.

    Args:
        cross_tab: Cross-tabulation table
    """
    print("\n" + "=" * 80, flush=True)
    print("CROSS-TABULATION TABLE (Weighted Population Counts)", flush=True)
    print("=" * 80, flush=True)
    print(cross_tab.to_string())
    print("=" * 80 + "\n", flush=True)
