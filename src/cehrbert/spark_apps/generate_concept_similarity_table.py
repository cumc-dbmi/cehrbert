"""This module provides functionality to extract patient event data from domain tables,.

compute information content and semantic similarity for concepts, and calculate concept
similarity scores.

Functions: extract_data: Extract data from specified domain tables. compute_information_content:
Compute the information content for concepts based on frequency.
compute_information_content_similarity: Compute the similarity between concepts based on
information content. compute_semantic_similarity: Compute the semantic similarity between concept
pairs. main: Main function to orchestrate the extraction, processing, and saving of concept
similarity data.
"""

import datetime
import logging
import os
from typing import List

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ..config.output_names import CONCEPT_SIMILARITY_PATH, QUALIFIED_CONCEPT_LIST_PATH
from ..const.common import CONCEPT, CONCEPT_ANCESTOR
from ..utils.spark_utils import (
    join_domain_tables,
    preprocess_domain_table,
    validate_table_names,
)


def extract_data(spark: SparkSession, input_folder: str, domain_table_list: List[str]):
    """
    Extract patient event data from the specified domain tables.

    Args:
        spark (SparkSession): The Spark session to use for processing.
        input_folder (str): Path to the input folder containing domain tables.
        domain_table_list (List[str]): List of domain table names to extract data from.

    Returns:
        DataFrame: A DataFrame containing extracted and processed patient event data.
    """
    domain_tables = []
    for domain_table_name in domain_table_list:
        domain_tables.append(
            preprocess_domain_table(spark, input_folder, domain_table_name)
        )
    patient_event = join_domain_tables(domain_tables)
    # Remove all concept_id records
    patient_event = patient_event.where("standard_concept_id <> 0")

    return patient_event


def compute_information_content(patient_event: DataFrame, concept_ancestor: DataFrame):
    """
    Calculate the information content using the frequency of each concept and the graph.

    :param patient_event:
    :param concept_ancestor:
    :return:
    """
    # Get the total count
    total_count = patient_event.distinct().count()
    # Count the frequency of each concept
    concept_frequency = patient_event.distinct().groupBy("standard_concept_id").count()
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

    return information_content


def compute_information_content_similarity(
    concept_pair: DataFrame, information_content: DataFrame, concept_ancestor: DataFrame
):
    """
    Compute the similarity between concept pairs based on their information content.

    Args:
        concept_pair (DataFrame): A DataFrame containing pairs of concepts.
        information_content (DataFrame): A DataFrame with information content for concepts.
        concept_ancestor (DataFrame): A DataFrame containing concept ancestor relationships.

    Returns:
        DataFrame: A DataFrame containing various similarity measures for concept pairs.
    """
    # Extract the pairs of concepts from the training data and join to the information content table
    information_content_concept_pair = (
        concept_pair.select("concept_id_1", "concept_id_2")
        .join(
            information_content,
            F.col("concept_id_1") == F.col("concept_id"),
            "left_outer",
        )
        .select(
            F.col("concept_id_1"),
            F.col("concept_id_2"),
            F.col("information_content").alias("information_content_1"),
        )
        .join(
            information_content,
            F.col("concept_id_2") == F.col("concept_id"),
            "left_outer",
        )
        .select(
            F.col("concept_id_1"),
            F.col("concept_id_2"),
            F.col("information_content_1"),
            F.col("information_content").alias("information_content_2"),
        )
    )

    # Join to get all the ancestors of concept_id_1
    concept_id_1_ancestor = information_content_concept_pair.join(
        concept_ancestor, F.col("concept_id_1") == F.col("descendant_concept_id")
    ).select("concept_id_1", "concept_id_2", "ancestor_concept_id")

    # Join to get all the ancestors of concept_id_2
    concept_id_2_ancestor = concept_pair.join(
        concept_ancestor, F.col("concept_id_2") == F.col("descendant_concept_id")
    ).select("concept_id_1", "concept_id_2", "ancestor_concept_id")

    # Compute the summed information content of all ancestors of concept_id_1 and concept_id_2
    union_sum = (
        concept_id_1_ancestor.union(concept_id_2_ancestor)
        .distinct()
        .join(information_content, F.col("ancestor_concept_id") == F.col("concept_id"))
        .groupBy("concept_id_1", "concept_id_2")
        .agg(F.sum("information_content").alias("ancestor_union_ic"))
    )

    # Compute the summed information content of common ancestors of concept_id_1 and concept_id_2
    intersection_sum = (
        concept_id_1_ancestor.intersect(concept_id_2_ancestor)
        .join(information_content, F.col("ancestor_concept_id") == F.col("concept_id"))
        .groupBy("concept_id_1", "concept_id_2")
        .agg(F.sum("information_content").alias("ancestor_intersection_ic"))
    )

    # Compute the information content and probability of the most informative common ancestor (MICA)
    mica_ancestor = (
        concept_id_1_ancestor.intersect(concept_id_2_ancestor)
        .join(information_content, F.col("ancestor_concept_id") == F.col("concept_id"))
        .groupBy("concept_id_1", "concept_id_2")
        .agg(
            F.max("information_content").alias("mica_information_content"),
            F.max("probability").alias("mica_probability"),
        )
    )

    # Join the MICA to pairs of concepts
    features = information_content_concept_pair.join(
        mica_ancestor,
        (
            information_content_concept_pair["concept_id_1"]
            == mica_ancestor["concept_id_1"]
        )
        & (
            information_content_concept_pair["concept_id_2"]
            == mica_ancestor["concept_id_2"]
        ),
        "left_outer",
    ).select(
        [
            information_content_concept_pair[f]
            for f in information_content_concept_pair.schema.fieldNames()
        ]
        + [F.col("mica_information_content"), F.col("mica_probability")]
    )

    # Compute the lin measure
    features = features.withColumn(
        "lin_measure",
        2
        * F.col("mica_information_content")
        / (F.col("information_content_1") * F.col("information_content_2")),
    )

    # Compute the jiang measure
    features = features.withColumn(
        "jiang_measure",
        1
        - (
            F.col("information_content_1")
            + F.col("information_content_2")
            - 2 * F.col("mica_information_content")
        ),
    )

    # Compute the information coefficient
    features = features.withColumn(
        "information_coefficient",
        F.col("lin_measure") * (1 - 1 / (1 + F.col("mica_information_content"))),
    )

    # Compute the relevance_measure
    features = features.withColumn(
        "relevance_measure", F.col("lin_measure") * (1 - F.col("mica_probability"))
    )

    # Join to get the summed information content of the common ancestors of concept_id_1 and
    # concept_id_2
    features = features.join(
        intersection_sum,
        (features["concept_id_1"] == intersection_sum["concept_id_1"])
        & (features["concept_id_2"] == intersection_sum["concept_id_2"]),
        "left_outer",
    ).select(
        [features[f] for f in features.schema.fieldNames()]
        + [F.col("ancestor_intersection_ic")]
    )

    # Join to get the summed information content of the common ancestors of concept_id_1 and
    # concept_id_2
    features = features.join(
        union_sum,
        (features["concept_id_1"] == union_sum["concept_id_1"])
        & (features["concept_id_2"] == union_sum["concept_id_2"]),
        "left_outer",
    ).select(
        [features[f] for f in features.schema.fieldNames()]
        + [F.col("ancestor_union_ic")]
    )

    # Compute the graph information content measure
    features = features.withColumn(
        "graph_ic_measure",
        F.col("ancestor_intersection_ic") / F.col("ancestor_union_ic"),
    )

    return features.select(
        [
            F.col("concept_id_1"),
            F.col("concept_id_2"),
            F.col("mica_information_content"),
            F.col("lin_measure"),
            F.col("jiang_measure"),
            F.col("information_coefficient"),
            F.col("relevance_measure"),
            F.col("graph_ic_measure"),
        ]
    )


def compute_semantic_similarity(spark, patient_event, concept, concept_ancestor):
    required_concept = (
        patient_event.distinct()
        .select("standard_concept_id")
        .join(concept, F.col("standard_concept_id") == F.col("concept_id"))
        .select("standard_concept_id", "domain_id")
    )

    concept_ancestor.createOrReplaceTempView("concept_ancestor")
    required_concept.createOrReplaceTempView("required_concept")

    concept_pair = spark.sql(
        """
        WITH concept_pair AS (
            SELECT
                c1.standard_concept_id AS concept_id_1,
                c2.standard_concept_id AS concept_id_2,
                c1.domain_id
            FROM required_concept AS c1
            JOIN required_concept AS c2
                ON c1.domain_id = c2.domain_id
            WHERE c1.standard_concept_id <> c2.standard_concept_id
        )
        SELECT
            cp.concept_id_1,
            cp.concept_id_2,
            ca_1.ancestor_concept_id AS common_ancestor_concept_id,
            ca_1.min_levels_of_separation AS distance_1,
            ca_2.min_levels_of_separation AS distance_2
        FROM concept_pair AS cp
        JOIN concept_ancestor AS ca_1
            ON cp.concept_id_1 = ca_1.descendant_concept_id
        JOIN concept_ancestor AS ca_2
            ON cp.concept_id_2 = ca_2.descendant_concept_id
        WHERE ca_1.ancestor_concept_id = ca_2.ancestor_concept_id
    """
    )

    # Find the root concepts
    root_concept = (
        concept_ancestor.groupBy("descendant_concept_id")
        .count()
        .where("count = 1")
        .withColumnRenamed("descendant_concept_id", "root_concept_id")
    )
    # Retrieve all ancestor descendant relationships for the root concepts
    root_concept_relationship = (
        root_concept.join(
            concept_ancestor,
            root_concept["root_concept_id"] == concept_ancestor["ancestor_concept_id"],
        )
        .select(
            concept_ancestor["ancestor_concept_id"],
            concept_ancestor["descendant_concept_id"],
            concept_ancestor["max_levels_of_separation"].alias("root_distance"),
        )
        .where("ancestor_concept_id <> descendant_concept_id")
    )

    # Join to get all root concepts and their corresponding root_distance
    concept_pair = concept_pair.join(
        root_concept_relationship,
        F.col("common_ancestor_concept_id") == F.col("descendant_concept_id"),
    ).select(
        "concept_id_1", "concept_id_2", "distance_1", "distance_2", "root_distance"
    )

    # Compute the semantic similarity
    concept_pair_similarity = concept_pair.withColumn(
        "semantic_similarity",
        2
        * F.col("root_distance")
        / (2 * F.col("root_distance") + F.col("distance_1") + F.col("distance_2")),
    )
    # Find the maximum semantic similarity
    concept_pair_similarity = concept_pair_similarity.groupBy(
        "concept_id_1", "concept_id_2"
    ).agg(F.max("semantic_similarity").alias("semantic_similarity"))

    return concept_pair_similarity


def main(
    input_folder: str,
    output_folder: str,
    domain_table_list: List[str],
    date_filter: str,
    include_concept_list: bool,
):
    """
    Main function to generate the concept similarity table.

    Args:
        input_folder (str): The path to the input folder containing raw data.
        output_folder (str): The path to the output folder to store the results.
        domain_table_list (List[str]): List of domain tables to process.
        date_filter (str): Date filter to apply to the data.
        include_concept_list (bool): Whether to include a filtered concept list.
    """

    spark = SparkSession.builder.appName(
        "Generate the concept similarity table"
    ).getOrCreate()

    logger = logging.getLogger(__name__)
    logger.info(
        "input_folder: %s\noutput_folder: %s\ndomain_table_list: %s\ndate_filter: "
        "%s\ninclude_concept_list: %s",
        input_folder,
        output_folder,
        domain_table_list,
        date_filter,
        include_concept_list,
    )

    concept = preprocess_domain_table(spark, input_folder, CONCEPT)
    concept_ancestor = preprocess_domain_table(spark, input_folder, CONCEPT_ANCESTOR)

    # Extract all data points from specified domains
    patient_event = extract_data(spark, input_folder, domain_table_list)

    # Calculate information content using unfiltered the patient event dataframe
    information_content = compute_information_content(patient_event, concept_ancestor)

    # Filter out concepts that are not required in the required concept_list
    if include_concept_list and patient_event:
        # Filter out concepts
        qualified_concepts = F.broadcast(
            preprocess_domain_table(spark, input_folder, QUALIFIED_CONCEPT_LIST_PATH)
        )

        patient_event = patient_event.join(
            qualified_concepts, "standard_concept_id"
        ).select("standard_concept_id")

    concept_pair_similarity = compute_semantic_similarity(
        spark, patient_event, concept, concept_ancestor
    )

    # Compute the information content based similarity scores
    concept_pair_ic_similarity = compute_information_content_similarity(
        concept_pair_similarity, information_content, concept_ancestor
    )

    concept_pair_similarity_columns = [
        concept_pair_similarity[f] for f in concept_pair_similarity.schema.fieldNames()
    ]
    concept_pair_ic_similarity_columns = [
        f
        for f in concept_pair_ic_similarity.schema.fieldNames()
        if "concept_id" not in f
    ]

    # Join two dataframes to get the final result
    concept_pair_similarity = concept_pair_similarity.join(
        concept_pair_ic_similarity,
        (
            concept_pair_similarity["concept_id_1"]
            == concept_pair_ic_similarity["concept_id_1"]
        )
        & (
            concept_pair_similarity["concept_id_2"]
            == concept_pair_ic_similarity["concept_id_2"]
        ),
    ).select(concept_pair_similarity_columns + concept_pair_ic_similarity_columns)

    concept_pair_similarity.write.mode("overwrite").parquet(
        os.path.join(output_folder, CONCEPT_SIMILARITY_PATH)
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Arguments for generate Concept Similarity Table"
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
    parser.add_argument(
        "--include_concept_list", dest="include_concept_list", action="store_true"
    )

    ARGS = parser.parse_args()

    main(
        ARGS.input_folder,
        ARGS.output_folder,
        ARGS.domain_table_list,
        ARGS.date_filter,
        ARGS.include_concept_list,
    )
