"""
Visualization module for score-based clustering analysis.

Provides functions to generate charts and visualizations for clustering results,
including pie charts, bar charts, and statistical summaries.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams

from ..utils.logger import get_logger

# Configure matplotlib for better-looking plots
rcParams["font.family"] = "DejaVu Sans"
rcParams["figure.figsize"] = (12, 6)
rcParams["axes.labelsize"] = 11
rcParams["xtick.labelsize"] = 10
rcParams["ytick.labelsize"] = 10
rcParams["legend.fontsize"] = 10


def prepare_visualization_data(statistics: dict[str, Any]) -> pd.DataFrame:
    """
    Prepare clustering statistics for visualization.

    Args:
        statistics: Cluster statistics dictionary from get_cluster_statistics()

    Returns:
        DataFrame with formatted statistics for plotting
    """
    data = []
    cluster_order = ["low", "middle", "high"]
    cluster_labels = {
        "low": "Low (< 482)",
        "middle": "Middle (482-606)",
        "high": "High (≥ 607)",
    }

    for level in cluster_order:
        if level in statistics:
            stats = statistics[level]
            data.append(
                {
                    "Cluster": cluster_labels[level],
                    "Sample Count": stats["sample_count"],
                    "Weighted Population": stats["weighted_count"],
                    "Population %": stats["population_percentage"],
                    "Mean Score": stats["mean_score"],
                    "Weighted Mean": stats["weighted_mean_score"],
                }
            )

    return pd.DataFrame(data)


def create_pie_chart(
    statistics: dict[str, Any], output_path: str = "clustering_pie_chart.png"
) -> str:
    """
    Create a pie chart showing population distribution across clusters.

    Args:
        statistics: Cluster statistics dictionary
        output_path: Output file path for PNG (default: clustering_pie_chart.png)

    Returns:
        Path to saved PNG file
    """
    logger = get_logger()

    # Prepare data
    df = prepare_visualization_data(statistics)

    # Create pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

    wedges, texts, autotexts = ax.pie(
        df["Population %"],
        labels=df["Cluster"],
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        textprops={"fontsize": 12, "weight": "bold"},
    )

    # Enhance text
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(11)
        autotext.set_weight("bold")

    ax.set_title(
        "Score-Based Clustering Distribution\n(Weighted Population %)",
        fontsize=14,
        weight="bold",
        pad=20,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Pie chart saved to: {output_path}")
    plt.close()

    return output_path


def create_bar_chart(
    statistics: dict[str, Any],
    metric: str = "Weighted Population",
    output_path: str = "clustering_bar_chart.png",
) -> str:
    """
    Create a bar chart for clustering statistics.

    Args:
        statistics: Cluster statistics dictionary
        metric: Which metric to display (default: Weighted Population)
               Options: "Sample Count", "Weighted Population", "Population %", "Mean Score"
        output_path: Output file path for PNG

    Returns:
        Path to saved PNG file
    """
    logger = get_logger()

    # Prepare data
    df = prepare_visualization_data(statistics)

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

    bars = ax.bar(
        df["Cluster"],
        df[metric],
        color=colors,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.8,
    )

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:,.0f}" if metric != "Population %" else f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
        )

    ax.set_xlabel("Score Cluster", fontsize=12, weight="bold")
    ax.set_ylabel(metric, fontsize=12, weight="bold")
    ax.set_title(f"Score-Based Clustering by {metric}", fontsize=14, weight="bold", pad=20)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Bar chart saved to: {output_path}")
    plt.close()

    return output_path


def create_comparison_chart(
    statistics: dict[str, Any], output_path: str = "clustering_comparison_chart.png"
) -> str:
    """
    Create a multi-metric comparison chart.

    Shows both sample count and weighted population side by side.

    Args:
        statistics: Cluster statistics dictionary
        output_path: Output file path for PNG

    Returns:
        Path to saved PNG file
    """
    logger = get_logger()

    # Prepare data
    df = prepare_visualization_data(statistics)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

    # Sample count chart
    bars1 = ax1.bar(
        df["Cluster"],
        df["Sample Count"],
        color=colors,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.8,
    )
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:,.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )
    ax1.set_ylabel("Sample Count", fontsize=11, weight="bold")
    ax1.set_title("Sample Size by Cluster", fontsize=12, weight="bold", pad=15)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Weighted population chart
    bars2 = ax2.bar(
        df["Cluster"],
        df["Weighted Population"],
        color=colors,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.8,
    )
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:,.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )
    ax2.set_ylabel("Weighted Population", fontsize=11, weight="bold")
    ax2.set_title("Weighted Population by Cluster", fontsize=12, weight="bold", pad=15)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle(
        "Score-Based Clustering: Sample vs. Weighted Population",
        fontsize=14,
        weight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Comparison chart saved to: {output_path}")
    plt.close()

    return output_path


def export_statistics_table(
    statistics: dict[str, Any], output_path: str = "clustering_statistics.csv"
) -> str:
    """
    Export clustering statistics to CSV file.

    Args:
        statistics: Cluster statistics dictionary
        output_path: Output file path for CSV

    Returns:
        Path to saved CSV file
    """
    logger = get_logger()

    # Prepare data
    df = prepare_visualization_data(statistics)

    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Statistics table exported to: {output_path}")

    return output_path


def create_all_visualizations(
    statistics: dict[str, Any], output_directory: str = "artifacts/visualizations"
) -> dict[str, str]:
    """
    Generate all available visualizations for clustering results.

    Args:
        statistics: Cluster statistics dictionary
        output_directory: Directory to save all visualizations

    Returns:
        Dictionary mapping visualization names to their file paths
    """
    logger = get_logger()

    # Create output directory if it doesn't exist
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate all visualizations
    results = {
        "pie_chart": create_pie_chart(
            statistics, str(output_path / "score_clustering_pie_chart.png")
        ),
        "bar_chart": create_bar_chart(
            statistics,
            metric="Weighted Population",
            output_path=str(output_path / "score_clustering_bar_chart.png"),
        ),
        "comparison_chart": create_comparison_chart(
            statistics,
            str(output_path / "score_clustering_comparison_chart.png"),
        ),
        "statistics_table": export_statistics_table(
            statistics,
            str(output_path / "score_clustering_statistics.csv"),
        ),
    }

    logger.info(f"All visualizations generated in: {output_directory}")
    print(f"\n✓ Visualizations saved to: {output_directory}", flush=True)
    for name, path in results.items():
        print(f"  - {name}: {path}", flush=True)

    return results
