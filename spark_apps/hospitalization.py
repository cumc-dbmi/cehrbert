from spark_apps.spark_app_base import ProspectiveCohortBuilderBase
from spark_apps.parameters import create_spark_args
from utils.common import *

COHORT_QUERY_TEMPLATE = """
SELECT
    v.person_id,
    v.first_visit_occurrence_id AS visit_occurrence_id,
    v.first_visit_start_date AS index_date,
    CASE 
        WHEN num_of_hospitalizations IS NULL THEN 0
        ELSE 1
    END AS label
FROM
(
    SELECT
        v.first_visit_occurrence_id,
        v.first_visit_start_date,
        v.person_id,
        SUM(CASE WHEN v3.visit_concept_id IN {visit_concept_ids} THEN 1 END) AS num_of_hospitalizations
    FROM global_temp.first_qualified_visit_occurrence AS v
    JOIN global_temp.visit_occurrence AS v3
        ON v.person_id = v3.person_id
    WHERE v.num_of_hospitalizations IS NULL
        AND DATEDIFF(v3.visit_start_date, v.first_visit_start_date) > {total_window}
    GROUP BY v.first_visit_occurrence_id, v.person_id, v.first_visit_start_date
) v
"""

FIRST_QUALIFIED_VISIT_QUERY_TEMPLATE = """
SELECT
    v.first_visit_occurrence_id,
    v.first_visit_start_date,
    v.person_id,
    SUM(CASE WHEN v.visit_concept_id IN {visit_concept_ids} THEN 1 END) AS num_of_hospitalizations
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
    JOIN global_temp.visit_occurrence AS v2
        ON v1.person_id = v2.person_id
    WHERE v1.first_visit_start_date < v2.visit_start_date
        AND v1.first_visit_occurrence_id <> v2.visit_occurrence_id
        AND DATEDIFF(v2.visit_start_date, v1.first_visit_start_date) <= {total_window}
        AND v1.first_visit_start_date >= '{date_lower_bound}'
        AND v1.first_visit_start_date <= '{date_upper_bound}'
) v
GROUP BY v.first_visit_occurrence_id, v.person_id, v.first_visit_start_date
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

    def preprocess_dependency(self):
        first_visit_query = FIRST_VISIT_QUERY_TEMPLATE.format(visit_concept_ids=VISIT_CONCEPT_IDS,
                                                              total_window=self.get_total_window())
        self.spark.sql(first_visit_query).createOrReplaceGlobalTempView(FIRST_VISIT_TABLE)

        first_qualified_visit_query = FIRST_QUALIFIED_VISIT_QUERY_TEMPLATE.format(
            visit_concept_ids=VISIT_CONCEPT_IDS,
            date_lower_bound=self._date_lower_bound,
            date_upper_bound=self._date_upper_bound,
            total_window=self.get_total_window())

        self.spark.sql(first_qualified_visit_query).createOrReplaceGlobalTempView(
            FIRST_QUALIFIED_VISIT_TABLE)

        cohort_query = COHORT_QUERY_TEMPLATE.format(visit_concept_ids=VISIT_CONCEPT_IDS,
                                                    total_window=self.get_total_window())
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
         age_lower_bound, age_upper_bound, observation_window, prediction_window,
         index_date_match_window):
    cohort_builder = HospitalizationCohortBuilder(cohort_name,
                                                  input_folder,
                                                  output_folder,
                                                  date_lower_bound,
                                                  date_upper_bound,
                                                  age_lower_bound,
                                                  age_upper_bound,
                                                  observation_window,
                                                  prediction_window,
                                                  index_date_match_window,
                                                  DOMAIN_TABLE_LIST,
                                                  DEPENDENCY_LIST)

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
         spark_args.index_date_match_window)
