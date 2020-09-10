import os
import argparse
import datetime

from pyspark.sql import SparkSession

import spark_apps.parameters as p
from utils.common import *

VISIT_CONCEPT_IDS = [9201, 262]

DOMAIN_TABLE_LIST = ['condition_occurrence', 'drug_exposure', 'procedure_occurrence']

PERSON = 'person'
VISIT_OCCURRENCE = 'visit_occurrence'
CONDITION_OCCURRENCE = 'condition_occurrence'


def main(input_folder, output_folder, date_filter, age_lower_bound, age_upper_bound,
         observation_window,
         prediction_window):
    spark = SparkSession.builder.appName('Generate Hospitalization Cohort').getOrCreate()

    total_window = observation_window + prediction_window
    patient_ehr_records = extract_ehr_records(spark, input_folder, DOMAIN_TABLE_LIST)

    person = spark.read.parquet(os.path.join(input_folder, PERSON))
    visit_occurrence = spark.read.parquet(os.path.join(input_folder, VISIT_OCCURRENCE))

    visit_occurrence_copy = visit_occurrence.rdd.toDF(visit_occurrence.schema) \
        .select(F.col('person_id'),
                F.col('visit_occurrence_id').alias('prev_visit_occurrence_id'),
                F.col('visit_concept_id').alias('prev_visit_concept_id'),
                F.col('visit_start_date').alias('prev_visit_start_date'))

    hospitalization = visit_occurrence \
        .select('visit_occurrence_id', 'person_id', 'visit_concept_id', 'visit_start_date') \
        .withColumn('visit_start_date', F.to_date('visit_start_date', 'yyyy-MM-dd')) \
        .where(F.col('visit_concept_id') != 0) \
        .where(F.col('visit_start_date') >= F.lit(date_filter).cast('date')) \
        .join(visit_occurrence_copy, 'person_id') \
        .where(F.col('visit_occurrence_id') != F.col('prev_visit_occurrence_id')) \
        .where(F.col('visit_start_date') > F.col('prev_visit_start_date')) \
        .withColumn('days_gap',
                    F.datediff(F.col('visit_start_date'), F.col('prev_visit_start_date'))) \
        .where(F.col('days_gap') >= prediction_window) \
        .where(F.col('days_gap') <= total_window)

    cohort = hospitalization \
        .where(F.col('visit_concept_id') != 0) \
        .groupBy('person_id', 'visit_occurrence_id', 'visit_concept_id', 'visit_start_date').agg(
        F.count('prev_visit_occurrence_id').alias('num_prior_visits')) \
        .withColumn('label', F.col('visit_concept_id').isin(VISIT_CONCEPT_IDS).cast('int')) \
        .withColumn('visit_order',
                    F.row_number().over(
                        W.partitionBy('person_id').orderBy(F.desc('label'), 'visit_start_date',
                                                           'visit_occurrence_id'))) \
        .where(F.col('visit_order') == 1) \
        .where(F.col('num_prior_visits') >= 3)

    cohort = cohort.join(person, 'person_id') \
        .withColumn('age', F.year('visit_start_date') - F.col('year_of_birth')) \
        .where(F.col('age').between(age_lower_bound, age_upper_bound)) \
        .select([F.col(field_name) for field_name in cohort.schema.fieldNames()] + [F.col('age'),
                                                                                    person[
                                                                                        'gender_concept_id'],
                                                                                    person[
                                                                                        'race_concept_id']])

    fh_cohort_ehr_records = patient_ehr_records.join(cohort,
                                                     patient_ehr_records['person_id'] == cohort[
                                                         'person_id']) \
        .where(patient_ehr_records['visit_occurrence_id'] != cohort['visit_occurrence_id']) \
        .where(
        patient_ehr_records['date'] <= F.date_sub(cohort['visit_start_date'], prediction_window)) \
        .where(patient_ehr_records['date'] >= F.date_sub(cohort['visit_start_date'], total_window)) \
        .select(patient_ehr_records['person_id'], patient_ehr_records['standard_concept_id'],
                patient_ehr_records['date'], patient_ehr_records['visit_occurrence_id'],
                patient_ehr_records['domain'])

    sequence_data = create_sequence_data(fh_cohort_ehr_records, None)

    cohort.select('person_id', 'label', 'age', 'gender_concept_id', 'race_concept_id') \
        .join(sequence_data, 'person_id') \
        .write.mode('overwrite').parquet(os.path.join(output_folder, p.hospitalization_data_path))


def valid_date(s):
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for generating mortality labels')
    parser.add_argument('-i',
                        '--input_folder',
                        dest='input_folder',
                        action='store',
                        help='The path for your input_folder where the sequence data is',
                        required=True)
    parser.add_argument('-o',
                        '--output_folder',
                        dest='output_folder',
                        action='store',
                        help='The path for your output_folder',
                        required=True)
    parser.add_argument('-f',
                        '--date_filter',
                        dest='date_filter',
                        action='store',
                        help='The path for your output_folder',
                        required=True,
                        type=valid_date)
    parser.add_argument('-l',
                        '--lower_bound',
                        dest='lower_bound',
                        action='store',
                        help='The age lower bound',
                        required=False,
                        type=int,
                        default=0)
    parser.add_argument('-u',
                        '--upper_bound',
                        dest='upper_bound',
                        action='store',
                        help='The age upper bound',
                        required=False,
                        type=int,
                        default=100)
    parser.add_argument('-ow',
                        '--observation_window',
                        dest='observation_window',
                        action='store',
                        help='The observation window in days for extracting features',
                        required=False,
                        type=int,
                        default=365)
    parser.add_argument('-pw',
                        '--prediction_window',
                        dest='prediction_window',
                        action='store',
                        help='The prediction window in days prior the index date',
                        required=False,
                        type=int,
                        default=180)

    ARGS = parser.parse_args()

    main(ARGS.input_folder,
         ARGS.output_folder,
         ARGS.date_filter,
         ARGS.lower_bound,
         ARGS.upper_bound,
         ARGS.observation_window,
         ARGS.prediction_window)
