import os
import argparse
import datetime

from pyspark.sql import SparkSession

import spark_apps.parameters as p
from utils.common import *

VISIT_CONCEPT_IDS = [9201, 9203, 262]

HEART_FAILURE_CONCEPTS = [45773075, 45766964, 45766167, 45766166, 45766165, 45766164, 44784442, 44784345, 44782733,
                          44782728, 44782719, 44782718, 44782713, 44782655, 44782428, 43530961, 43530643, 43530642,
                          43022068, 43022054, 43021842, 43021841, 43021840, 43021826, 43021825, 43021736, 43021735,
                          43020657, 43020421, 40486933, 40482857, 40481043, 40481042, 40480603, 40480602, 40479576,
                          40479192, 37311948, 37309625, 37110330, 36717359, 36716748, 36716182, 36713488, 36712929,
                          36712928, 36712927, 35615055, 4327205, 4311437, 4307356, 4284562, 4273632, 4267800, 4264636,
                          4259490, 4242669, 4233424, 4233224, 4229440, 4215802, 4215446, 4206009, 4205558, 4199500,
                          4195892, 4195785, 4193236, 4185565, 4177493, 4172864, 4142561, 4141124, 4139864, 4138307,
                          4124705, 4111554, 4108245, 4108244, 4103448, 4079695, 4079296, 4071869, 4030258, 4023479,
                          4014159, 4009047, 4004279, 3184320, 764877, 764876, 764874, 764873, 764872, 764871, 762003,
                          762002, 444101, 444031, 443587, 443580, 442310, 439846, 439698, 439696, 439694, 319835,
                          316994, 316139, 314378, 312927]

DOMAIN_TABLE_LIST = ['condition_occurrence', 'drug_exposure', 'procedure_occurrence']

PERSON = 'person'
VISIT_OCCURRENCE = 'visit_occurrence'
CONDITION_OCCURRENCE = 'condition_occurrence'


def main(input_folder, output_folder, date_filter, age_lower_bound, age_upper_bound, observation_window,
         prediction_window):
    spark = SparkSession.builder.appName('Generate Heart Failure Cohort').getOrCreate()

    total_window = observation_window + prediction_window
    patient_ehr_records = extract_ehr_records(spark, input_folder, DOMAIN_TABLE_LIST)

    person = spark.read.parquet(os.path.join(input_folder, PERSON))
    visit_occurrence = spark.read.parquet(os.path.join(input_folder, VISIT_OCCURRENCE))
    condition_occurrence = spark.read.parquet(os.path.join(input_folder, CONDITION_OCCURRENCE))

    visits = visit_occurrence.where(F.col('visit_concept_id').isin(VISIT_CONCEPT_IDS))
    heart_failure_conditions = condition_occurrence.where(F.col('condition_concept_id').isin(HEART_FAILURE_CONCEPTS))

    positive_hf_cases = visits.join(heart_failure_conditions, heart_failure_conditions['visit_occurrence_id'] == visits[
        'visit_occurrence_id']).select(
        visits['visit_occurrence_id'], visits['person_id'], visits['visit_start_date']).distinct()

    hf_person_ids = positive_hf_cases.select(F.col('person_id').alias('positive_person_id')).distinct()

    qualified_positive_hf_cases = positive_hf_cases \
        .withColumn('visit_start_date', F.to_date('visit_start_date', 'yyyy-MM-dd')) \
        .withColumn('earliest_visit_start_date',
                    F.first('visit_start_date').over(W.partitionBy('person_id').orderBy('visit_start_date'))) \
        .withColumn('earliest_visit_occurrence_id',
                    F.first('visit_occurrence_id').over(W.partitionBy('person_id').orderBy('visit_start_date'))) \
        .withColumn('num_of_diagnosis', F.count('visit_occurrence_id').over(W.partitionBy('person_id'))) \
        .where(F.col('num_of_diagnosis') >= 3) \
        .select(F.col('person_id'),
                F.col('earliest_visit_start_date').alias('visit_start_date'),
                F.col('earliest_visit_occurrence_id').alias('visit_occurrence_id')) \
        .where(F.col('visit_start_date') >= date_filter) \
        .distinct().withColumn('label', F.lit(1))

    negative_hf_cases = visits.join(hf_person_ids, F.col('person_id') == F.col('positive_person_id'), 'left') \
        .where(F.col('positive_person_id').isNull()) \
        .select(visits['visit_occurrence_id'], visits['person_id'], visits['visit_start_date']).distinct() \
        .withColumn('visit_start_date', F.to_date('visit_start_date', 'yyyy-MM-dd')) \
        .withColumn('latest_visit_start_date', F.first('visit_start_date')
                    .over(W.partitionBy('person_id').orderBy(F.desc('visit_start_date')))) \
        .withColumn('latest_visit_occurrence_id', F.first('visit_occurrence_id')
                    .over(W.partitionBy('person_id').orderBy(F.desc('visit_start_date')))) \
        .where(F.col('visit_start_date') <= F.date_sub(F.col('latest_visit_start_date'), prediction_window)) \
        .where(F.col('visit_start_date') >= F.date_sub(F.col('latest_visit_start_date'), total_window)) \
        .withColumn('num_of_visits', F.count('person_id').over(W.partitionBy('person_id'))) \
        .where(F.col('num_of_visits') >= 3) \
        .select(F.col('person_id'),
                F.col('latest_visit_start_date').alias('visit_start_date'),
                F.col('latest_visit_occurrence_id').alias('visit_occurrence_id')) \
        .where(F.col('visit_start_date') >= date_filter) \
        .distinct().withColumn('label', F.lit(0))

    fh_cohort = qualified_positive_hf_cases.union(negative_hf_cases)

    fh_cohort = fh_cohort.join(person, 'person_id') \
        .withColumn('age', F.year('visit_start_date') - F.col('year_of_birth')) \
        .where(F.col('age').between(age_lower_bound, age_upper_bound)) \
        .select([F.col(field_name) for field_name in fh_cohort.schema.fieldNames()] + [F.col('age'),
                                                                                       person['gender_concept_id'],
                                                                                       person['race_concept_id']])

    fh_cohort_ehr_records = patient_ehr_records.join(fh_cohort,
                                                     patient_ehr_records['person_id'] == fh_cohort['person_id']) \
        .where(patient_ehr_records['visit_occurrence_id'] != fh_cohort['visit_occurrence_id']) \
        .where(patient_ehr_records['date'] <= F.date_sub(fh_cohort['visit_start_date'], prediction_window)) \
        .where(patient_ehr_records['date'] >= F.date_sub(fh_cohort['visit_start_date'], total_window)) \
        .select(patient_ehr_records['person_id'], patient_ehr_records['standard_concept_id'],
                patient_ehr_records['date'], patient_ehr_records['visit_occurrence_id'],
                patient_ehr_records['domain'])

    sequence_data = create_sequence_data(fh_cohort_ehr_records, date_filter)

    fh_cohort.select('person_id', 'label', 'age', 'gender_concept_id', 'race_concept_id') \
        .join(sequence_data, 'person_id') \
        .write.mode('overwrite').parquet(os.path.join(output_folder, p.heart_failure_data_path))


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
