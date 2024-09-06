import os

from pyspark.sql import SparkSession

from ..const.common import CONCEPT, MEASUREMENT, REQUIRED_MEASUREMENT
from ..utils.spark_utils import F, W, argparse, preprocess_domain_table


def main(input_folder, output_folder, num_of_numeric_labs, num_of_categorical_labs):
    spark = SparkSession.builder.appName("Generate required labs").getOrCreate()

    # Load measurement as a dataframe in pyspark
    measurement = preprocess_domain_table(spark, input_folder, MEASUREMENT)
    concept = preprocess_domain_table(spark, input_folder, CONCEPT)

    # Create the local measurement view
    measurement.createOrReplaceTempView("measurement")

    # Create the local concept view
    concept.createOrReplaceTempView("concept")

    popular_labs = spark.sql(
        """
        SELECT
            m.measurement_concept_id,
            c.concept_name,
            COUNT(*) AS freq,
            SUM(CASE WHEN m.value_as_number IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*) AS numeric_percentage,
            SUM(CASE WHEN m.value_as_concept_id IS NOT NULL AND m.value_as_concept_id <> 0 THEN 1 ELSE 0 END) / COUNT(*) AS categorical_percentage
        FROM measurement AS m
        JOIN concept AS c
            ON m.measurement_concept_id = c.concept_id
        WHERE m.measurement_concept_id <> 0
        GROUP BY m.measurement_concept_id, c.concept_name
        ORDER BY COUNT(*) DESC
    """
    )

    # Cache the dataframe for faster computation in the below transformations
    popular_labs.cache()

    popular_numeric_labs = (
        popular_labs.withColumn("is_numeric", F.col("numeric_percentage") >= 0.5)
        .where("is_numeric")
        .withColumn("rn", F.row_number().over(W.orderBy(F.desc("freq"))))
        .where(F.col("rn") <= num_of_numeric_labs)
        .drop("rn")
    )

    popular_categorical_labs = (
        popular_labs.withColumn("is_categorical", F.col("categorical_percentage") >= 0.5)
        .where("is_categorical")
        .withColumn("is_numeric", ~F.col("is_categorical"))
        .withColumn("rn", F.row_number().over(W.orderBy(F.desc("freq"))))
        .where(F.col("rn") <= num_of_categorical_labs)
        .drop("is_categorical")
        .drop("rn")
    )

    popular_numeric_labs.unionAll(popular_categorical_labs).write.mode("overwrite").parquet(
        os.path.join(output_folder, REQUIRED_MEASUREMENT)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for generate " "required labs to be included"
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
        "--num_of_numeric_labs",
        dest="num_of_numeric_labs",
        action="store",
        type=int,
        default=100,
        help="The top most popular numeric labs to be included",
        required=False,
    )
    parser.add_argument(
        "--num_of_categorical_labs",
        dest="num_of_categorical_labs",
        action="store",
        type=int,
        default=100,
        help="The top most popular categorical labs to be included",
        required=False,
    )

    ARGS = parser.parse_args()

    main(
        ARGS.input_folder,
        ARGS.output_folder,
        ARGS.num_of_numeric_labs,
        ARGS.num_of_categorical_labs,
    )
