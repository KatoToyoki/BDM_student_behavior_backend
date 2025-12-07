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

    # Assign fixed colors based on cluster semantic meaning
    # Green = Proactive, Yellow = Average, Red = Disengaged
    color_map = {
        "Proactive Learners": "#95E1D3",  # Green - Positive
        "Average Learners": "#FDB750",  # Yellow - Neutral
        "Disengaged Learners": "#FF6B6B",  # Red - Concerning
    }
    colors = [color_map[cluster] for cluster in df["Cluster"]]

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

    # Assign fixed colors based on cluster semantic meaning
    # Green = Proactive, Yellow = Average, Red = Disengaged
    color_map = {
        "Proactive Learners": "#95E1D3",  # Green - Positive
        "Average Learners": "#FDB750",  # Yellow - Neutral
        "Disengaged Learners": "#FF6B6B",  # Red - Concerning
    }
    colors = [color_map[cluster] for cluster in df["Cluster"]]

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

    # Determine subtitle based on metric type
    subtitle = (
        "(Weighted Population Statistics)"
        if "%" in metric or "Population" in metric
        else "(Unweighted Sample Count)"
    )

    ax.set_title(
        f"Attitude Clustering: {metric}\n{subtitle}",
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

    # Calculate percentage distribution for comparison
    df["Sample Count %"] = df["Sample Count"] / df["Sample Count"].sum() * 100
    df["Weighted Population %"] = df["Weighted Population"] / df["Weighted Population"].sum() * 100

    # Create comparison chart
    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(df))
    width = 0.35

    bars1 = ax.bar(
        [i - width / 2 for i in x],
        df["Sample Count %"],
        width,
        label="Sample Count Distribution",
        color="#4ECDC4",
        edgecolor="black",
        linewidth=1.5,
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x],
        df["Weighted Population %"],
        width,
        label="Weighted Population Distribution",
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

    ax.set_ylabel("Distribution (%)", fontsize=12, weight="bold")
    ax.set_xlabel("Attitude Cluster", fontsize=12, weight="bold")
    ax.set_title(
        "Attitude Clustering: Sample vs Weighted Population Distribution\n(Each distribution sums to 100%)",
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


def create_missing_value_chart(
    missing_stats: dict[str, Any],
    output_path: str = "visualizations/attitude_missing_values.png",
) -> str:
    """
    Create a bar chart showing missing value rates for each variable.

    Args:
        missing_stats: Missing value statistics from prepare_attitude_data()
        output_path: Output file path for PNG

    Returns:
        Path to saved PNG file
    """
    logger = get_logger()

    # Prepare data
    data = []
    for var_name, stats in missing_stats.items():
        # Get readable variable name
        from ..analysis.attitude_clustering import ATTITUDE_DIMENSIONS

        readable_name = ATTITUDE_DIMENSIONS.get(var_name, var_name)
        data.append(
            {
                "Variable": readable_name,
                "Variable_Code": var_name,
                "Missing_Rate": stats["missing_rate"],
                "Missing_Count": stats["missing_count"],
                "Valid_Count": stats["valid_count"],
            }
        )

    df = pd.DataFrame(data)

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [
        "#FF6B6B" if x > 10 else "#FDB750" if x > 5 else "#95E1D3" for x in df["Missing_Rate"]
    ]

    bars = ax.bar(
        df["Variable"], df["Missing_Rate"], color=colors, edgecolor="black", linewidth=1.5
    )

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )

    ax.set_ylabel("Missing Rate (%)", fontsize=12, weight="bold")
    ax.set_xlabel("Attitude Dimension", fontsize=12, weight="bold")
    ax.set_title(
        "Missing Value Rates by Attitude Dimension",
        fontsize=14,
        weight="bold",
        pad=20,
    )
    ax.set_ylim(0, max(df["Missing_Rate"].max() * 1.2, 5))
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    # Save with 300 DPI
    full_path = Path(output_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(full_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Missing value chart saved to {output_path}")
    return str(full_path)


def create_sample_loss_chart(
    sample_loss: dict[str, Any],
    weighted_loss: dict[str, Any],
    output_path: str = "visualizations/attitude_sample_loss.png",
) -> str:
    """
    Create a comparison chart showing sample loss (unweighted vs weighted).

    Args:
        sample_loss: Sample loss statistics
        weighted_loss: Weighted population loss statistics
        output_path: Output file path for PNG

    Returns:
        Path to saved PNG file
    """
    logger = get_logger()

    # Prepare data
    categories = ["Retained", "Removed"]
    sample_values = [
        sample_loss["cleaned_count"],
        sample_loss["removed_count"],
    ]
    weighted_values = [
        weighted_loss["cleaned_weighted"],
        weighted_loss["removed_weighted"],
    ]

    # Create side-by-side bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Unweighted sample
    colors1 = ["#95E1D3", "#FF6B6B"]
    bars1 = ax1.bar(categories, sample_values, color=colors1, edgecolor="black", linewidth=1.5)

    for bar, val in zip(bars1, sample_values):
        height = bar.get_height()
        percentage = val / sample_loss["original_count"] * 100
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:,}\n({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )

    ax1.set_ylabel("Sample Count", fontsize=12, weight="bold")
    ax1.set_title(
        f"Unweighted Sample Loss\nTotal Loss: {sample_loss['loss_rate']:.2f}%",
        fontsize=13,
        weight="bold",
        pad=15,
    )
    ax1.set_ylim(0, sample_loss["original_count"] * 1.15)

    # Weighted population
    colors2 = ["#95E1D3", "#FF6B6B"]
    bars2 = ax2.bar(categories, weighted_values, color=colors2, edgecolor="black", linewidth=1.5)

    for bar, val in zip(bars2, weighted_values):
        height = bar.get_height()
        percentage = val / weighted_loss["original_weighted"] * 100
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:,.0f}\n({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )

    ax2.set_ylabel("Weighted Population", fontsize=12, weight="bold")
    ax2.set_title(
        f"Weighted Population Loss\nTotal Loss: {weighted_loss['weighted_loss_rate']:.2f}%",
        fontsize=13,
        weight="bold",
        pad=15,
    )
    ax2.set_ylim(0, weighted_loss["original_weighted"] * 1.15)

    plt.suptitle(
        "Sample Loss Analysis: Listwise Deletion",
        fontsize=15,
        weight="bold",
        y=1.02,
    )
    plt.tight_layout()

    # Save with 300 DPI
    full_path = Path(output_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(full_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Sample loss chart saved to {output_path}")
    return str(full_path)


def export_missing_value_table(
    missing_stats: dict[str, Any],
    sample_loss: dict[str, Any],
    weighted_loss: dict[str, Any],
    output_path: str = "visualizations/attitude_missing_values.csv",
) -> str:
    """
    Export missing value and sample loss statistics to CSV.

    Args:
        missing_stats: Missing value statistics
        sample_loss: Sample loss statistics
        weighted_loss: Weighted population loss statistics
        output_path: Output file path for CSV

    Returns:
        Path to saved CSV file
    """
    logger = get_logger()

    # Variable-level missing rates
    from ..analysis.attitude_clustering import ATTITUDE_DIMENSIONS

    data = []
    for var_name, stats in missing_stats.items():
        readable_name = ATTITUDE_DIMENSIONS.get(var_name, var_name)
        data.append(
            {
                "Variable": readable_name,
                "Variable_Code": var_name,
                "Total_Count": stats["valid_count"] + stats["missing_count"],
                "Valid_Count": stats["valid_count"],
                "Missing_Count": stats["missing_count"],
                "Missing_Rate_%": round(stats["missing_rate"], 2),
            }
        )

    df = pd.DataFrame(data)

    # Add summary rows
    summary_data = pd.DataFrame(
        [
            {
                "Variable": "--- SAMPLE LOSS SUMMARY ---",
                "Variable_Code": "",
                "Total_Count": "",
                "Valid_Count": "",
                "Missing_Count": "",
                "Missing_Rate_%": "",
            },
            {
                "Variable": "Original Sample",
                "Variable_Code": "",
                "Total_Count": sample_loss["original_count"],
                "Valid_Count": sample_loss["cleaned_count"],
                "Missing_Count": sample_loss["removed_count"],
                "Missing_Rate_%": round(sample_loss["loss_rate"], 2),
            },
            {
                "Variable": "--- WEIGHTED POPULATION LOSS ---",
                "Variable_Code": "",
                "Total_Count": "",
                "Valid_Count": "",
                "Missing_Count": "",
                "Missing_Rate_%": "",
            },
            {
                "Variable": "Original Population",
                "Variable_Code": "",
                "Total_Count": int(weighted_loss["original_weighted"]),
                "Valid_Count": int(weighted_loss["cleaned_weighted"]),
                "Missing_Count": int(weighted_loss["removed_weighted"]),
                "Missing_Rate_%": round(weighted_loss["weighted_loss_rate"], 2),
            },
        ]
    )

    df_final = pd.concat([df, summary_data], ignore_index=True)

    # Save to CSV
    full_path = Path(output_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(full_path, index=False)

    logger.info(f"Missing value table exported to {output_path}")
    return str(full_path)
