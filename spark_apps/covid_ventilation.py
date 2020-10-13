from pyspark.sql import DataFrame

from spark_apps.spark_app_base import AbstractCohortBuilderBase
from spark_apps.parameters import create_spark_args

from utils.common import *

PERSON = 'person'
VISIT_OCCURRENCE = 'visit_occurrence'
CONDITION_OCCURRENCE = 'condition_occurrence'
VENT = 'vent'
MEASUREMENT = 'measurement'
CONCEPT = 'concept'
DEATH = 'death'
DEPENDENCY_LIST = [PERSON, VISIT_OCCURRENCE, VENT, MEASUREMENT, CONDITION_OCCURRENCE, DEATH, CONCEPT]
DOMAIN_TABLE_LIST = ['condition_occurrence', 'drug_exposure', 'procedure_occurrence']

COHORT_QUERY = """
WITH covid_patients AS 
(
    SELECT
        c.person_id,
        MIN(c.index_date) AS index_date
    FROM
    (
        SELECT 
            m.person_id,
            MIN(measurement_date) AS index_date
        FROM global_temp.measurement AS m
        JOIN global_temp.concept AS c
            ON m.value_as_concept_id = c.concept_id
        WHERE m.measurement_concept_id IN (723475,723479,706178,723473,723474,586515,706177,706163,706180,706181)
            AND c.concept_name IN ('Detected', 'Positve')
        GROUP BY person_id

        UNION

        SELECT 
            co.person_id,
            MIN(co.condition_start_date) AS index_date
        FROM global_temp.condition_occurrence AS co
        WHERE co.condition_concept_id IN (4100065, 37311061)
        GROUP BY co.person_id
    ) c
    GROUP BY c.person_id
)

SELECT
    pa.person_id,
    p.gender_concept_id,
    p.race_concept_id,
    YEAR(pa.index_date) - p.year_of_birth AS age,
    pa.index_date,
    CAST(ISNOTNULL(COALESCE(vent.person_id)) AS int) AS label
FROM covid_patients AS pa
JOIN global_temp.person AS p
    ON pa.person_id = p.person_id
LEFT JOIN global_temp.vent AS vent
    ON vent.person_id = pa.person_id
"""


class CovidVentilationCohortBuilder(AbstractCohortBuilderBase):

    def preprocess_dependencies(self):
        pass

    def create_cohort(self):
        return self.spark.sql(COHORT_QUERY)

    def extract_ehr_records_for_cohort(self, cohort: DataFrame):
        ehr_records = extract_ehr_records(self.spark,
                                          self._input_folder,
                                          self._ehr_table_list)

        visit_occurrence = self._dependency_dict[VISIT_OCCURRENCE]

        ehr_records = ehr_records.join(visit_occurrence, 'visit_occurrence_id') \
            .select([ehr_records[field_name] for field_name in ehr_records.schema.fieldNames()]
                    + [visit_occurrence['visit_concept_id']])

        cohort_ehr_records = ehr_records.join(cohort, 'person_id') \
            .select([ehr_records[field_name] for field_name in ehr_records.schema.fieldNames()])

        return create_sequence_data(cohort_ehr_records, None)


def main(cohort_name, input_folder, output_folder, date_lower_bound, date_upper_bound,
         age_lower_bound, age_upper_bound, observation_window, prediction_window,
         index_date_match_window):
    cohort_builder = CovidVentilationCohortBuilder(cohort_name,
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
