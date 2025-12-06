"""
Visualization module for cross-dimensional analysis of attitude and score clustering.

Provides functions to generate various charts showing the relationship between
attitude clusters and score-based clusters using pandas and seaborn.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams

from ..utils.logger import get_logger

# Configure matplotlib and seaborn for better-looking plots
rcParams["font.family"] = "DejaVu Sans"
rcParams["figure.figsize"] = (14, 8)
rcParams["axes.labelsize"] = 11
rcParams["xtick.labelsize"] = 10
rcParams["ytick.labelsize"] = 10
rcParams["legend.fontsize"] = 10

sns.set_style("whitegrid")
sns.set_palette("husl")


def prepare_cross_visualization_data(cross_tab: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare cross-tabulation data for visualization.

    Args:
        cross_tab: Cross-tabulation table

    Returns:
        DataFrame with normalized data for visualization
    """
    # Create a copy for visualization
    df_viz = cross_tab.copy()

    # Add row and column totals
    df_viz["Total"] = df_viz.sum(axis=1)
    df_viz.loc["Total"] = df_viz.sum()

    return df_viz


def create_grouped_bar_chart(
    cross_tab: pd.DataFrame,
    output_path: str = "visualizations/attitude_score_grouped_bar_chart.png",
) -> str:
    """
    Create grouped bar chart showing score distribution across attitude clusters.

    Args:
        cross_tab: Cross-tabulation table
        output_path: Output file path

    Returns:
        Path to saved PNG file
    """
    logger = get_logger()

    fig, ax = plt.subplots(figsize=(12, 7))

    # Prepare data for grouped bars
    x = np.arange(len(cross_tab.index))
    width = 0.25
    colors = ["#4ECDC4", "#95E1D3", "#FDB750"]

    for i, col in enumerate(cross_tab.columns):
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset,
            cross_tab[col],
            width,
            label=col,
            color=colors[i],
            edgecolor="black",
            linewidth=1.2,
        )
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height):,}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xlabel("Score Cluster", fontsize=12, weight="bold")
    ax.set_ylabel("Weighted Population Count", fontsize=12, weight="bold")
    ax.set_title(
        "Attitude-Score Relationship: Grouped Distribution\n(Weighted Population by Score and Attitude)",
        fontsize=13,
        weight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(cross_tab.index, rotation=0)
    ax.legend(title="Attitude Cluster", fontsize=10, title_fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x / 1e6)}M"))

    plt.tight_layout()
    full_path = Path(output_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(full_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Grouped bar chart saved to {output_path}")
    return str(full_path)


def create_stacked_bar_chart(
    cross_tab: pd.DataFrame,
    output_path: str = "visualizations/attitude_score_stacked_bar_chart.png",
) -> str:
    """
    Create stacked bar chart showing attitude composition within each score cluster.

    Args:
        cross_tab: Cross-tabulation table
        output_path: Output file path

    Returns:
        Path to saved PNG file
    """
    logger = get_logger()

    fig, ax = plt.subplots(figsize=(12, 7))

    # Calculate percentages for stacked chart
    cross_tab_pct = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100

    cross_tab_pct.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=["#4ECDC4", "#95E1D3", "#FDB750"],
        edgecolor="black",
        linewidth=1.2,
    )

    ax.set_xlabel("Score Cluster", fontsize=12, weight="bold")
    ax.set_ylabel("Percentage (%)", fontsize=12, weight="bold")
    ax.set_title(
        "Attitude-Score Relationship: Stacked Distribution\n(Percentage Composition of Attitude Clusters within Each Score)",
        fontsize=13,
        weight="bold",
        pad=20,
    )
    ax.legend(title="Attitude Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xticklabels(cross_tab_pct.index, rotation=45, ha="right")
    ax.set_ylim(0, 100)

    # Add percentage labels
    for container in ax.containers:
        ax.bar_label(container, label_type="center", fontsize=9, weight="bold")

    plt.tight_layout()
    full_path = Path(output_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(full_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Stacked bar chart saved to {output_path}")
    return str(full_path)


def create_heatmap(
    cross_tab: pd.DataFrame,
    output_path: str = "visualizations/attitude_score_heatmap.png",
) -> str:
    """
    Create heatmap showing normalized values of cross-tabulation.

    Args:
        cross_tab: Cross-tabulation table
        output_path: Output file path

    Returns:
        Path to saved PNG file
    """
    logger = get_logger()

    fig, ax = plt.subplots(figsize=(12, 7))

    # Normalize for heatmap (0-1 scale)
    cross_tab_norm = (cross_tab - cross_tab.min().min()) / (
        cross_tab.max().max() - cross_tab.min().min()
    )

    sns.heatmap(
        cross_tab_norm,
        annot=cross_tab,
        fmt=".0f",
        cmap="YlOrRd",
        cbar_kws={"label": "Normalized Intensity"},
        ax=ax,
        linewidths=1,
        linecolor="gray",
    )

    ax.set_xlabel("Attitude Cluster", fontsize=12, weight="bold")
    ax.set_ylabel("Score Cluster", fontsize=12, weight="bold")
    ax.set_title(
        "Attitude-Score Relationship: Heatmap\n(Weighted Population Counts)",
        fontsize=13,
        weight="bold",
        pad=20,
    )

    plt.tight_layout()
    full_path = Path(output_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(full_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Heatmap saved to {output_path}")
    return str(full_path)


def create_normalized_heatmap(
    cross_tab: pd.DataFrame,
    output_path: str = "visualizations/attitude_score_heatmap_normalized.png",
) -> str:
    """
    Create normalized heatmap showing row percentages (distribution across attitude within each score).

    Args:
        cross_tab: Cross-tabulation table
        output_path: Output file path

    Returns:
        Path to saved PNG file
    """
    logger = get_logger()

    fig, ax = plt.subplots(figsize=(12, 7))

    # Normalize by rows (percentage within each score cluster)
    cross_tab_pct = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100

    sns.heatmap(
        cross_tab_pct,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        cbar_kws={"label": "Percentage (%)"},
        ax=ax,
        linewidths=1,
        linecolor="gray",
        vmin=0,
        vmax=100,
    )

    ax.set_xlabel("Attitude Cluster", fontsize=12, weight="bold")
    ax.set_ylabel("Score Cluster", fontsize=12, weight="bold")
    ax.set_title(
        "Attitude-Score Relationship: Normalized Heatmap\n(% Distribution of Attitudes within Each Score Cluster)",
        fontsize=13,
        weight="bold",
        pad=20,
    )

    plt.tight_layout()
    full_path = Path(output_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(full_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Normalized heatmap saved to {output_path}")
    return str(full_path)


def create_all_score_attitude_visualizations(
    cross_tab: pd.DataFrame,
    output_directory: str = "visualizations",
) -> dict[str, str]:
    """
    Generate all score-attitude cross-analysis visualizations.

    Args:
        cross_tab: Cross-tabulation table
        output_directory: Directory for output files

    Returns:
        Dictionary mapping visualization names to file paths
    """
    logger = get_logger()

    # Ensure output directory exists
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating score-attitude cross-analysis visualizations...")

    visualizations = {
        "grouped_bar_chart": create_grouped_bar_chart(
            cross_tab,
            str(output_dir / "score_attitude_grouped_bar_chart.png"),
        ),
        "stacked_bar_chart": create_stacked_bar_chart(
            cross_tab,
            str(output_dir / "score_attitude_stacked_bar_chart.png"),
        ),
        "heatmap": create_heatmap(
            cross_tab,
            str(output_dir / "score_attitude_heatmap.png"),
        ),
        "normalized_heatmap": create_normalized_heatmap(
            cross_tab,
            str(output_dir / "score_attitude_heatmap_normalized.png"),
        ),
    }

    logger.info(f"Generated {len(visualizations)} score-attitude cross-analysis visualizations")

    return visualizations
