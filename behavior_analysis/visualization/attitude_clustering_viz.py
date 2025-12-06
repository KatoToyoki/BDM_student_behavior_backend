"""
Visualization module for attitude-based clustering analysis.

Provides functions to generate charts and visualizations for attitude clustering results,
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


def prepare_attitude_visualization_data(statistics: dict[str, Any]) -> pd.DataFrame:
    """
    Prepare attitude clustering statistics for visualization.

    Args:
        statistics: Cluster statistics dictionary from get_attitude_statistics()

    Returns:
        DataFrame with formatted statistics for plotting
    """
    data = []
    cluster_labels = [
        "Proactive Learners",
        "Average Learners",
        "Disengaged Learners",
    ]

    for label in cluster_labels:
        if label in statistics:
            stats = statistics[label]
            data.append(
                {
                    "Cluster": label,
                    "Sample Count": stats["sample_count"],
                    "Weighted Population": stats["weighted_count"],
                    "Population %": stats["population_percentage"],
                }
            )

    return pd.DataFrame(data)


def create_attitude_pie_chart(
    statistics: dict[str, Any],
    output_path: str = "visualizations/attitude_clustering_pie_chart.png",
) -> str:
    """
    Create a pie chart showing attitude cluster distribution.

    Args:
        statistics: Cluster statistics dictionary
        output_path: Output file path for PNG

    Returns:
        Path to saved PNG file
    """
    logger = get_logger()

    # Prepare data
    df = prepare_attitude_visualization_data(statistics)

    # Create pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#95E1D3", "#FDB750", "#FF6B6B"]

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
        "Attitude-Based Clustering Distribution\n(Weighted Population %)",
        fontsize=14,
        weight="bold",
        pad=20,
    )

    plt.tight_layout()

    # Save with 300 DPI
    full_path = Path(output_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(full_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Pie chart saved to {output_path}")
    return str(full_path)


def create_attitude_bar_chart(
    statistics: dict[str, Any],
    metric: str = "Population %",
    output_path: str = "visualizations/attitude_clustering_bar_chart.png",
) -> str:
    """
    Create a bar chart for attitude clustering metrics.

    Args:
        statistics: Cluster statistics dictionary
        metric: Metric to display (default: "Population %")
        output_path: Output file path for PNG

    Returns:
        Path to saved PNG file
    """
    logger = get_logger()

    # Prepare data
    df = prepare_attitude_visualization_data(statistics)

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#95E1D3", "#FDB750", "#FF6B6B"]

    bars = ax.bar(df["Cluster"], df[metric], color=colors, edgecolor="black", linewidth=1.5)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}" if metric == "Population %" else f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
        )

    ax.set_ylabel(metric, fontsize=12, weight="bold")
    ax.set_xlabel("Attitude Cluster", fontsize=12, weight="bold")
    ax.set_title(
        f"Attitude Clustering: {metric}\n(Weighted Statistics)",
        fontsize=14,
        weight="bold",
        pad=20,
    )
    ax.set_ylim(0, df[metric].max() * 1.15)  # Add 15% padding at top
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    # Save with 300 DPI
    full_path = Path(output_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(full_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Bar chart saved to {output_path}")
    return str(full_path)


def create_attitude_comparison_chart(
    statistics: dict[str, Any],
    output_path: str = "visualizations/attitude_clustering_comparison_chart.png",
) -> str:
    """
    Create a comparison chart of sample count vs weighted population.

    Args:
        statistics: Cluster statistics dictionary
        output_path: Output file path for PNG

    Returns:
        Path to saved PNG file
    """
    logger = get_logger()

    # Prepare data
    df = prepare_attitude_visualization_data(statistics)

    # Normalize for comparison
    df["Sample Count (normalized)"] = df["Sample Count"] / df["Sample Count"].max() * 100
    df["Weighted Population (normalized)"] = (
        df["Weighted Population"] / df["Weighted Population"].max() * 100
    )

    # Create comparison chart
    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(df))
    width = 0.35

    bars1 = ax.bar(
        [i - width / 2 for i in x],
        df["Sample Count (normalized)"],
        width,
        label="Sample Count",
        color="#4ECDC4",
        edgecolor="black",
        linewidth=1.5,
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x],
        df["Weighted Population (normalized)"],
        width,
        label="Weighted Population",
        color="#FF6B6B",
        edgecolor="black",
        linewidth=1.5,
    )

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.0f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                weight="bold",
            )

    ax.set_ylabel("Normalized Value (%)", fontsize=12, weight="bold")
    ax.set_xlabel("Attitude Cluster", fontsize=12, weight="bold")
    ax.set_title(
        "Attitude Clustering: Sample Count vs Weighted Population\n(Normalized to 100%)",
        fontsize=14,
        weight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["Cluster"], rotation=15, ha="right")
    ax.legend(fontsize=11, loc="upper right")
    ax.set_ylim(0, 120)

    plt.tight_layout()

    # Save with 300 DPI
    full_path = Path(output_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(full_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Comparison chart saved to {output_path}")
    return str(full_path)


def export_attitude_statistics_table(
    statistics: dict[str, Any],
    output_path: str = "visualizations/attitude_clustering_statistics.csv",
) -> str:
    """
    Export attitude clustering statistics to CSV.

    Args:
        statistics: Cluster statistics dictionary
        output_path: Output file path for CSV

    Returns:
        Path to saved CSV file
    """
    logger = get_logger()

    # Prepare data
    df = prepare_attitude_visualization_data(statistics)

    # Save to CSV
    full_path = Path(output_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(full_path, index=False)

    logger.info(f"Statistics table exported to {output_path}")
    return str(full_path)


def create_all_attitude_visualizations(
    statistics: dict[str, Any], output_directory: str = "visualizations"
) -> dict[str, str]:
    """
    Generate all visualizations for attitude clustering.

    Args:
        statistics: Cluster statistics dictionary
        output_directory: Directory for output files

    Returns:
        Dictionary mapping visualization names to file paths
    """
    logger = get_logger()

    # Ensure output directory exists
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating attitude clustering visualizations...")

    # Generate all visualizations
    visualizations = {
        "pie_chart": create_attitude_pie_chart(
            statistics,
            str(output_dir / "attitude_clustering_pie_chart.png"),
        ),
        "population_bar_chart": create_attitude_bar_chart(
            statistics,
            "Population %",
            str(output_dir / "attitude_clustering_population_bar_chart.png"),
        ),
        "sample_count_bar_chart": create_attitude_bar_chart(
            statistics,
            "Sample Count",
            str(output_dir / "attitude_clustering_sample_count_bar_chart.png"),
        ),
        "comparison_chart": create_attitude_comparison_chart(
            statistics,
            str(output_dir / "attitude_clustering_comparison_chart.png"),
        ),
        "statistics_table": export_attitude_statistics_table(
            statistics,
            str(output_dir / "attitude_clustering_statistics.csv"),
        ),
    }

    logger.info(f"Generated {len(visualizations)} visualizations")

    return visualizations
