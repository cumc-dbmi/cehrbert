from spark_apps.spark_app_base import ProspectiveCohortBuilderBase
from spark_apps.parameters import create_spark_args
from utils.spark_utils import *

COHORT_QUERY_TEMPLATE = """
SELECT
    v.person_id,
    v.first_visit_occurrence_id AS visit_occurrence_id,
    v.first_visit_start_date AS index_date,
    CAST(num_of_hospitalizations > 0 AS INT) AS label
FROM
(
    SELECT
        v.first_visit_occurrence_id,
        v.first_visit_start_date,
        v.person_id,
        SUM(CASE WHEN v3.visit_concept_id IN {visit_concept_ids} THEN 1 ELSE 0 END) AS num_of_hospitalizations
    FROM global_temp.first_qualified_visit_occurrence AS v
    LEFT JOIN global_temp.visit_occurrence AS v3
        ON v.person_id = v3.person_id 
            AND DATEDIFF(v3.visit_start_date, v.first_visit_start_date) BETWEEN {total_window} AND {prediction_window}
    GROUP BY v.first_visit_occurrence_id, v.person_id, v.first_visit_start_date
) v
"""

FIRST_QUALIFIED_VISIT_QUERY_TEMPLATE = """
SELECT
    v.first_visit_occurrence_id,
    v.first_visit_start_date,
    v.person_id
FROM
(
    SELECT
        v.first_visit_occurrence_id,
        v.first_visit_start_date,
        v.person_id,
        SUM(CASE WHEN v.visit_concept_id IN {visit_concept_ids} THEN 1 ELSE 0 END) AS num_of_hospitalizations
    FROM
    (
        SELECT DISTINCT
            v1.person_id,
            v1.first_visit_concept_id,
            v1.first_visit_start_date,
            v1.first_visit_occurrence_id,
            v2.visit_concept_id,
            v2.visit_start_date,
            v2.visit_occurrence_id
        FROM global_temp.first_visit_occurrence AS v1
        LEFT JOIN global_temp.visit_occurrence AS v2
            ON v1.person_id = v2.person_id
                AND v1.first_visit_start_date < v2.visit_start_date
                AND v1.first_visit_occurrence_id <> v2.visit_occurrence_id
                AND DATEDIFF(v2.visit_start_date, v1.first_visit_start_date) <= {total_window}
    ) v
    GROUP BY v.first_visit_occurrence_id, v.person_id, v.first_visit_start_date
) v
WHERE v.num_of_hospitalizations = 0
"""

FIRST_VISIT_QUERY_TEMPLATE = """
SELECT
    *
FROM
(
    SELECT DISTINCT
        v1.person_id,
        FIRST(visit_concept_id) OVER (PARTITION BY person_id ORDER BY visit_start_date, visit_occurrence_id) AS first_visit_concept_id,
        FIRST(visit_start_date) OVER (PARTITION BY person_id ORDER BY visit_start_date, visit_occurrence_id) AS first_visit_start_date,
        FIRST(visit_occurrence_id) OVER (PARTITION BY person_id ORDER BY visit_start_date, visit_occurrence_id) AS first_visit_occurrence_id
    FROM global_temp.visit_occurrence AS v1
) v
WHERE v.first_visit_concept_id NOT IN {visit_concept_ids} 
    AND v.first_visit_start_date >= '{date_lower_bound}'
    AND v.first_visit_start_date <= '{date_upper_bound}'
"""

COHORT_TABLE = 'cohort'
FIRST_VISIT_TABLE = 'first_visit_occurrence'
FIRST_QUALIFIED_VISIT_TABLE = 'first_qualified_visit_occurrence'

VISIT_CONCEPT_IDS = (9201, 262)

DOMAIN_TABLE_LIST = ['condition_occurrence', 'drug_exposure', 'procedure_occurrence']

PERSON = 'person'
VISIT_OCCURRENCE = 'visit_occurrence'
DEPENDENCY_LIST = [PERSON, VISIT_OCCURRENCE]


class HospitalizationCohortBuilder(ProspectiveCohortBuilderBase):

    def preprocess_dependencies(self):
        first_visit_query = FIRST_VISIT_QUERY_TEMPLATE.format(
            date_lower_bound=self._date_lower_bound,
            date_upper_bound=self._date_upper_bound,
            visit_concept_ids=VISIT_CONCEPT_IDS)
        self.spark.sql(first_visit_query).createOrReplaceGlobalTempView(FIRST_VISIT_TABLE)

        # The qualifying patients can't have any hospitalization record before observation_window
        # plus hold_off_window
        total_window = self._observation_window + self._hold_off_window
        prediction_window = total_window + self._prediction_window

        first_qualified_visit_query = FIRST_QUALIFIED_VISIT_QUERY_TEMPLATE.format(
            visit_concept_ids=VISIT_CONCEPT_IDS,
            total_window=total_window)

        self.spark.sql(first_qualified_visit_query).createOrReplaceGlobalTempView(
            FIRST_QUALIFIED_VISIT_TABLE)

        cohort_query = COHORT_QUERY_TEMPLATE.format(visit_concept_ids=VISIT_CONCEPT_IDS,
                                                    total_window=total_window,
                                                    prediction_window=prediction_window)
        cohort = self.spark.sql(cohort_query)

        cohort.createOrReplaceGlobalTempView(COHORT_TABLE)

        self._dependency_dict[COHORT_TABLE] = cohort

    def create_incident_cases(self):
        cohort = self._dependency_dict[COHORT_TABLE]
        person = self._dependency_dict[PERSON]

        incident_cases = cohort.where(F.col('label') == 1) \
            .join(person, 'person_id') \
            .withColumn('age', F.year('index_date') - F.col('year_of_birth')) \
            .select(F.col('person_id'),
                    F.col('age'),
                    F.col('gender_concept_id'),
                    F.col('race_concept_id'),
                    F.col('year_of_birth'),
                    F.col('index_date'),
                    F.col('visit_occurrence_id'),
                    F.col('label')).distinct()

        return incident_cases

    def create_control_cases(self):
        cohort = self._dependency_dict[COHORT_TABLE]
        person = self._dependency_dict[PERSON]

        control_cases = cohort.where(F.col('label') == 0) \
            .join(person, 'person_id') \
            .withColumn('age', F.year('index_date') - F.col('year_of_birth')) \
            .select(F.col('person_id'),
                    F.col('age'),
                    F.col('gender_concept_id'),
                    F.col('race_concept_id'),
                    F.col('year_of_birth'),
                    F.col('index_date'),
                    F.col('visit_occurrence_id'),
                    F.col('label')).distinct()

        return control_cases


def main(cohort_name, input_folder, output_folder, date_lower_bound, date_upper_bound,
         age_lower_bound, age_upper_bound, observation_window, prediction_window, hold_off_window,
         index_date_match_window, include_visit_type, is_feature_concept_frequency,
         is_roll_up_concept):
    cohort_builder = HospitalizationCohortBuilder(cohort_name,
                                                  input_folder,
                                                  output_folder,
                                                  date_lower_bound,
                                                  date_upper_bound,
                                                  age_lower_bound,
                                                  age_upper_bound,
                                                  observation_window,
                                                  prediction_window,
                                                  hold_off_window,
                                                  index_date_match_window,
                                                  DOMAIN_TABLE_LIST,
                                                  DEPENDENCY_LIST,
                                                  True,
                                                  include_visit_type,
                                                  is_feature_concept_frequency,
                                                  is_roll_up_concept)

    cohort_builder.build()


if __name__ == '__main__':
    spark_args = create_spark_args()

    main(spark_args.cohort_name,
         spark_args.input_folder,
         spark_args.output_folder,
         spark_args.date_lower_bound,
         spark_args.date_upper_bound,
         spark_args.lower_bound,
         spark_args.upper_bound,
         spark_args.observation_window,
         spark_args.prediction_window,
         spark_args.hold_off_window,
         spark_args.index_date_match_window,
         spark_args.include_visit_type,
         spark_args.is_feature_concept_frequency,
         spark_args.is_roll_up_concept)
