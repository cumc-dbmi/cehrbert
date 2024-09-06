"""
This module generates a qualified concept list by processing patient event data across various.

domain tables (e.g., condition_occurrence, procedure_occurrence, drug_exposure) and applying a
patient frequency filter to retain concepts linked to a minimum number of patients.

Key Functions:
    - preprocess_domain_table: Preprocesses domain tables to prepare for event extraction.
    - join_domain_tables: Joins multiple domain tables into a unified DataFrame.
    - main: Coordinates the entire process of reading domain tables, applying frequency filters,
      and saving the qualified concept list.

Command-line Arguments:
    - input_folder: Directory containing the input data.
    - output_folder: Directory where the qualified concept list will be saved.
    - min_num_of_patients: Minimum number of patients linked to a concept for it to be included.
    - with_drug_rollup: Boolean flag indicating whether drug concept rollups should be applied.
"""

import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from ..config.output_names import QUALIFIED_CONCEPT_LIST_PATH
from ..const.common import MEASUREMENT
from ..utils.spark_utils import join_domain_tables, preprocess_domain_table

DOMAIN_TABLE_LIST = ["condition_occurrence", "procedure_occurrence", "drug_exposure"]


def main(
    input_folder, output_folder, min_num_of_patients, with_drug_rollup: bool = True
):
    """
    Main function to generate a qualified concept list based on patient event data from multiple.

    domain tables.

    Args:
        input_folder (str): The directory where the input data is stored.
        output_folder (str): The directory where the output (qualified concept list) will be saved.
        min_num_of_patients (int): Minimum number of patients that a concept must be linked to for
        nclusion.
        with_drug_rollup (bool): If True, applies drug rollup logic to the drug_exposure domain.

    The function processes patient event data across various domain tables, excludes low-frequency
    concepts, and saves the filtered concepts to a specified output folder.
    """
    spark = SparkSession.builder.appName("Generate concept list").getOrCreate()

    domain_tables = []
    # Exclude measurement from domain_table_list if exists because we need to process measurement
    # in a different way
    for domain_table_name in DOMAIN_TABLE_LIST:
        if domain_table_name != MEASUREMENT:
            domain_tables.append(
                preprocess_domain_table(
                    spark,
                    input_folder,
                    domain_table_name,
                    with_drug_rollup=with_drug_rollup,
                )
            )

    # Union all domain table records
    patient_events = join_domain_tables(domain_tables)

    # Filter out concepts that are linked to less than 100 patients
    qualified_concepts = (
        patient_events.where("visit_occurrence_id IS NOT NULL")
        .groupBy("standard_concept_id")
        .agg(F.countDistinct("person_id").alias("freq"))
        .where(F.col("freq") >= min_num_of_patients)
    )

    qualified_concepts.write.mode("overwrite").parquet(
        os.path.join(output_folder, QUALIFIED_CONCEPT_LIST_PATH)
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Arguments for generate concept list to be included"
    )
    parser.add_argument(
        "-i",
        "--input_folder",
        dest="input_folder",
        action="store",
        help="The path for your input_folder where the raw data is",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        dest="output_folder",
        action="store",
        help="The path for your output_folder",
        required=True,
    )
    parser.add_argument(
        "--min_num_of_patients",
        dest="min_num_of_patients",
        action="store",
        type=int,
        default=0,
        help="Min no.of patients linked to concepts to be included",
        required=False,
    )
    parser.add_argument(
        "--with_drug_rollup", dest="with_drug_rollup", action="store_true"
    )

    ARGS = parser.parse_args()

    main(
        ARGS.input_folder,
        ARGS.output_folder,
        ARGS.min_num_of_patients,
        ARGS.with_drug_rollup,
    )
