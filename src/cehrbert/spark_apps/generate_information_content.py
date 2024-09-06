"""
This module generates an information content table based on a list of domain tables from OMOP data.

It processes patient event data, calculates the frequency of each concept, and computes information
conten using the concept ancestor hierarchy. The results are written to a specified output path.

Key Functions:
    - preprocess_domain_table: Preprocess the domain tables for analysis.
    - join_domain_tables: Join multiple domain tables to generate a unified patient event table.
    - main: Orchestrates the process of reading input data, calculating concept frequencies,
      and generating the information content table.

Command-line Arguments:
    - input_folder: The folder containing the raw OMOP domain data.
    - output_folder: The folder where the results will be stored.
    - domain_table_list: A list of OMOP domain tables to include in the analysis.
    - date_filter: Optional date filter for processing the data.
"""

import datetime
import logging
import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from ..config.output_names import INFORMATION_CONTENT_DATA_PATH
from ..const.common import CONCEPT_ANCESTOR
from ..utils.spark_utils import (
    join_domain_tables,
    preprocess_domain_table,
    validate_table_names,
)


def main(input_folder, output_folder, domain_table_list, date_filter):
    """Create the information content table.

    Keyword arguments:
    domain_tables -- the array containing the OMOP domain tables except visit_occurrence
    concept_id_frequency_output -- the path for writing the concept frequency output

    This function creates the information content table based on the given domain tables
    """

    spark = SparkSession.builder.appName(
        "Generate the information content table"
    ).getOrCreate()

    logger = logging.getLogger(__name__)
    logger.info(
        "input_folder: %s\noutput_folder: %s\ndomain_table_list: %s\ndate_filter: %s",
        input_folder,
        output_folder,
        domain_table_list,
        date_filter,
    )

    concept_ancestor = preprocess_domain_table(spark, input_folder, CONCEPT_ANCESTOR)
    domain_tables = []
    for domain_table_name in domain_table_list:
        domain_tables.append(
            preprocess_domain_table(spark, input_folder, domain_table_name)
        )

    patient_events = join_domain_tables(domain_tables)

    # Remove all concept_id records
    patient_events = patient_events.where("standard_concept_id <> 0")

    # Get the total count
    total_count = patient_events.distinct().count()

    # Count the frequency of each concept
    concept_frequency = patient_events.distinct().groupBy("standard_concept_id").count()

    # left join b/w descendent_concept_id and the standard_concept_id in the concept freq table
    freq_df = (
        concept_frequency.join(
            concept_ancestor,
            F.col("descendant_concept_id") == F.col("standard_concept_id"),
        )
        .groupBy("ancestor_concept_id")
        .sum("count")
        .withColumnRenamed("ancestor_concept_id", "concept_id")
        .withColumnRenamed("sum(count)", "count")
    )

    # Calculate information content for each concept
    information_content = freq_df.withColumn(
        "information_content", (-F.log(F.col("count") / total_count))
    ).withColumn("probability", F.col("count") / total_count)

    information_content.write.mode("overwrite").parquet(
        os.path.join(output_folder, INFORMATION_CONTENT_DATA_PATH)
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Arguments for generate training data for Bert"
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
        "-tc",
        "--domain_table_list",
        dest="domain_table_list",
        nargs="+",
        action="store",
        help="The list of domain tables you want to download",
        type=validate_table_names,
        required=True,
    )
    parser.add_argument(
        "-d",
        "--date_filter",
        dest="date_filter",
        type=lambda s: datetime.datetime.strptime(s, "%Y-%m-%d"),
        action="store",
        required=False,
        default="2018-01-01",
    )

    ARGS = parser.parse_args()

    main(
        ARGS.input_folder, ARGS.output_folder, ARGS.domain_table_list, ARGS.date_filter
    )
