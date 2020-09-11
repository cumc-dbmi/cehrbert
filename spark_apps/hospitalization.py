from spark_apps.spark_app_base import CohortBuilderBase
from spark_apps.parameters import create_spark_args
from utils.common import *


COHORT_NAME = 'hospitalization'
VISIT_CONCEPT_IDS = [9201, 262]

DOMAIN_TABLE_LIST = ['condition_occurrence', 'drug_exposure', 'procedure_occurrence']

PERSON = 'person'
VISIT_OCCURRENCE = 'visit_occurrence'
DEPENDENCY_LIST = [PERSON, VISIT_OCCURRENCE]

FIRST_VISIT_TABLE_NAME = 'first_visit_occurrence'
FIRST_VISIT_QUERY = """
SELECT DISTINCT
    v.person_id,
    v.visit_occurrence_id,
    v.visit_concept_id,
    v.visit_start_date,
    v.next_visit_occurrence_id,
    v.next_visit_concept_id,
    v.next_visit_start_date,
    DATEDIFF(v.next_visit_start_date, v.visit_start_date) AS days_gap
FROM
(
    SELECT
        person_id,
        visit_occurrence_id,
        visit_concept_id,
        visit_start_date,
        LEAD(visit_start_date) OVER (PARTITION BY person_id ORDER BY visit_start_date) AS next_visit_start_date,
        LEAD(visit_occurrence_id) OVER (PARTITION BY person_id ORDER BY visit_start_date) AS next_visit_occurrence_id,
        LEAD(visit_concept_id) OVER (PARTITION BY person_id ORDER BY visit_start_date) AS next_visit_concept_id,
        FIRST(visit_start_date) OVER (PARTITION BY person_id ORDER BY visit_start_date) AS first_visit_start_date,
        FIRST(visit_occurrence_id) OVER (PARTITION BY person_id ORDER BY visit_start_date) AS first_visit_occurrence_id
    FROM global_temp.visit_occurrence
) v
WHERE v.visit_occurrence_id = v.first_visit_occurrence_id 
    AND v.next_visit_occurrence_id IS NOT NULL
    AND v.first_visit_start_date >= '{date_filter}'
"""


class HospitalizationCohortBuilder(CohortBuilderBase):

    def preprocess_dependency(self):
        first_visit_table = self.spark.sql(FIRST_VISIT_QUERY.format(date_filter=self._date_filter))
        first_visit_table.createOrReplaceGlobalTempView(FIRST_VISIT_TABLE_NAME)
        self._dependency_dict[FIRST_VISIT_TABLE_NAME] = first_visit_table

    def create_incident_cases(self):
        first_visit_table = self._dependency_dict[FIRST_VISIT_TABLE_NAME]
        person = self._dependency_dict[PERSON]
        incident_cases = first_visit_table.where(
            F.col('days_gap').between(self._prediction_window, self._observation_window)) \
            .where(F.col('visit_concept_id').isin(VISIT_CONCEPT_IDS))

        incident_cases.join(person, 'person_id') \
            .withColumn('age', F.year('next_visit_start_date') - F.col('year_of_birth')) \
            .select(F.col('person_id'),
                    F.col('age'),
                    F.col('gender_concept_id'),
                    F.col('race_concept_id'),
                    F.col('year_of_birth'),
                    F.col('next_visit_start_date').alias('visit_start_date'),
                    F.col('next_visit_occurrence_id').alias('visit_occurrence_id'),
                    F.lit(1).alias('label')).distinct()

        return incident_cases

    def create_control_cases(self):
        first_visit_table = self._dependency_dict[FIRST_VISIT_TABLE_NAME]
        person = self._dependency_dict[PERSON]
        control_cases = first_visit_table.where(
            F.col('days_gap').between(self._prediction_window, self._observation_window)) \
            .where(~F.col('visit_concept_id').isin(VISIT_CONCEPT_IDS))

        control_cases.join(person, 'person_id') \
            .withColumn('age', F.year('next_visit_start_date') - F.col('year_of_birth')) \
            .select(F.col('person_id'),
                    F.col('age'),
                    F.col('gender_concept_id'),
                    F.col('race_concept_id'),
                    F.col('year_of_birth'),
                    F.col('next_visit_start_date').alias('visit_start_date'),
                    F.col('next_visit_occurrence_id').alias('visit_occurrence_id'),
                    F.lit(0).alias('label')).distinct()

        return control_cases


def main(input_folder, output_folder, date_filter, age_lower_bound, age_upper_bound,
         observation_window,
         prediction_window):

    cohort_builder = HospitalizationCohortBuilder(COHORT_NAME,
                                                  input_folder,
                                                  output_folder,
                                                  date_filter,
                                                  age_lower_bound,
                                                  age_upper_bound,
                                                  observation_window,
                                                  prediction_window,
                                                  DOMAIN_TABLE_LIST,
                                                  DEPENDENCY_LIST)
    cohort_builder.build()


if __name__ == '__main__':
    spark_args = create_spark_args()

    main(spark_args.input_folder,
         spark_args.output_folder,
         spark_args.date_filter,
         spark_args.lower_bound,
         spark_args.upper_bound,
         spark_args.observation_window,
         spark_args.prediction_window)
