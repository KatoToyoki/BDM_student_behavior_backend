"""
Score-Attitude Cross-Dimensional Analysis Test Module.

Tests and demonstrates the relationship analysis between attitude and score clustering.
"""

from .analysis.score_attitude_cross_analysis import (
    create_cross_tabulation,
    export_cross_tabulation,
    perform_chi_square_test,
    print_chi_square_report,
    print_cross_tabulation,
)
from .analysis.score_clustering import (
    add_cluster_labels as add_score_labels,
    categorize_score,
)
from .analysis.attitude_clustering import (
    add_attitude_labels,
)
from .config import load_config
from .data.spark_manager import SparkSessionManager
from .utils.logger import get_logger
from .visualization.score_attitude_cross_analysis_viz import (
    create_all_score_attitude_visualizations,
)


def test_score_attitude_cross_analysis() -> None:
    """
    Test score-attitude cross-dimensional analysis.

    This function loads student data, performs both clustering analyses,
    creates cross-tabulation, performs chi-square test, and generates visualizations.
    """
    logger = get_logger()

    print("\n" + "=" * 70, flush=True)
    print("CROSS-DIMENSIONAL ANALYSIS TEST", flush=True)
    print("=" * 70 + "\n", flush=True)

    try:
        # Initialize Spark session
        config = load_config()
        spark_manager = SparkSessionManager(config.spark)
        spark = spark_manager.get_session()
        logger.info("Spark session initialized for cross-analysis test")

        # Load student data
        student_parquet_path = config.get_parquet_path("student")
        print(f"Loading student data from: {student_parquet_path}", flush=True)

        student_df = spark.read.parquet(student_parquet_path)
        print(f"✓ Loaded: {student_df.count():,} records\n", flush=True)

        # Prepare score clustering
        print("Preparing score clustering...", flush=True)
        from pyspark.sql import functions as f

        df_with_score = student_df.withColumn(
            "score_cluster", categorize_score(score_column="PV1MATH")
        )
        df_with_score_labels = add_score_labels(
            df_with_score, score_column="PV1MATH", cluster_column="score_cluster"
        )
        print("✓ Score clustering completed\n", flush=True)

        # Prepare attitude clustering
        print("Preparing attitude clustering...", flush=True)
        from .analysis.attitude_clustering import (
            prepare_attitude_data,
            create_attitude_features,
            perform_attitude_clustering,
        )

        df_clean = prepare_attitude_data(df_with_score_labels)
        df_with_features = create_attitude_features(df_clean)
        df_clustered = perform_attitude_clustering(df_with_features, num_clusters=3)
        df_with_attitude_labels = add_attitude_labels(df_clustered)
        print("✓ Attitude clustering completed\n", flush=True)

        # Create cross-tabulation
        print("Creating cross-tabulation table...", flush=True)
        cross_tab = create_cross_tabulation(
            df_with_attitude_labels,
            score_column="score_cluster",
            attitude_column="attitude_cluster",
            weight_column="W_FSTUWT",
            score_label_column="score_cluster",
            attitude_label_column="attitude_label",
        )
        print("✓ Cross-tabulation created\n", flush=True)

        # Display cross-tabulation
        print_cross_tabulation(cross_tab)

        # Export cross-tabulation
        print("Exporting cross-tabulation...", flush=True)
        artifact_path = config.get_artifact_path("visualizations")
        export_cross_tabulation(
            cross_tab,
            str(artifact_path + "/attitude_score_crosstab.csv"),
        )
        print("✓ Cross-tabulation exported\n", flush=True)

        # Perform chi-square test
        print("Performing chi-square test...", flush=True)
        chi_results = perform_chi_square_test(cross_tab)
        print("✓ Chi-square test completed\n", flush=True)

        # Display chi-square results
        print_chi_square_report(chi_results)

        # Generate visualizations
        print("=" * 70, flush=True)
        print("Generating visualizations...", flush=True)
        print("=" * 70 + "\n", flush=True)

        visualizations = create_all_score_attitude_visualizations(
            cross_tab, artifact_path
        )

        print("✓ Visualizations generated:", flush=True)
        for viz_name, viz_path in visualizations.items():
            print(f"  - {viz_name}: {viz_path}", flush=True)

        print("\n" + "=" * 70, flush=True)
        print(
            "✓ SCORE-ATTITUDE CROSS-DIMENSIONAL ANALYSIS TEST COMPLETED SUCCESSFULLY",
            flush=True,
        )
        print("=" * 70 + "\n", flush=True)

    except Exception as e:
        logger.error(
            f"Score-attitude cross-dimensional analysis test failed: {e}", exc_info=True
        )
        print(f"\n✗ Error: {e}\n", flush=True)
        raise
