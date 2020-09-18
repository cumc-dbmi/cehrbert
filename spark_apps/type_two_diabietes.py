from pyspark.sql import DataFrame

from spark_apps.spark_app_base import AbstractCohortBuilderBase, cohort_validator
from spark_apps.parameters import create_spark_args

from utils.common import *

COHORT_QUERY_TEMPLATE = """
WITH person_ids_to_exclude AS 
(
    SELECT DISTINCT
        co.person_id
    FROM global_temp.condition_occurrence AS co 
    JOIN global_temp.{diabetes_exclusion_concepts} AS e
        ON co.condition_concept_id = e.concept_id
)
SELECT
    c.person_id,
    c.index_date,
    c.visit_occurrence_id,
    YEAR(c.index_date) - p.year_of_birth AS age,
    p.gender_concept_id,
    p.race_concept_id
FROM
(
    SELECT DISTINCT
        vo.person_id,
        FIRST(DATE(vo.visit_start_date)) OVER (PARTITION BY co.person_id 
            ORDER BY DATE(vo.visit_start_date), vo.visit_occurrence_id) AS index_date,
        FIRST(vo.visit_occurrence_id) OVER (PARTITION BY co.person_id 
            ORDER BY DATE(vo.visit_start_date), vo.visit_occurrence_id) AS visit_occurrence_id
    FROM global_temp.condition_occurrence AS co
    JOIN global_temp.{diabetes_inclusion_concepts} AS ie
        ON co.condition_concept_id = ie.concept_id
    JOIN global_temp.visit_occurrence AS vo
        ON co.visit_occurrence_id = vo.visit_occurrence_id
    LEFT JOIN person_ids_to_exclude AS e 
        ON co.person_id = e.person_id
    WHERE e.person_id IS NULL
) c
JOIN global_temp.person AS p 
    ON c.person_id = p.person_id
WHERE YEAR(c.index_date) - p.year_of_birth >= {age_lower_bound}
    AND c.index_date >= '{date_lower_bound}'
"""

DIABETES_INCLUSION = [443238, 201820, 442793]
DIABETES_EXCLUSION = [40484648, 201254, 435216, 4058243]

DOMAIN_TABLE_LIST = ['condition_occurrence', 'drug_exposure', 'procedure_occurrence']
DEPENDENCY_LIST = ['person', 'condition_occurrence', 'visit_occurrence',
                   'concept', 'concept_ancestor']

DIABETES_INCLUSION_TABLE = 'diabetes_inclusion_concepts'
DIABETES_EXCLUSION_TABLE = 'diabetes_exclusion_concepts'


class TypeTwoDiabetesCohortBuilder(AbstractCohortBuilderBase):
    cohort_required_columns = ['person_id', 'index_date']

    def preprocess_dependencies(self):
        diabetes_inclusion_concepts = get_descendant_concept_ids(self.spark, DIABETES_INCLUSION)
        diabetes_exclusion_concepts = get_descendant_concept_ids(self.spark, DIABETES_EXCLUSION)
        self._dependency_dict[DIABETES_INCLUSION_TABLE] = diabetes_inclusion_concepts
        self._dependency_dict[DIABETES_EXCLUSION_TABLE] = diabetes_exclusion_concepts
        diabetes_inclusion_concepts.createOrReplaceGlobalTempView(DIABETES_INCLUSION_TABLE)
        diabetes_exclusion_concepts.createOrReplaceGlobalTempView(DIABETES_EXCLUSION_TABLE)

    @cohort_validator('cohort_required_columns')
    def create_cohort(self):
        cohort_query = COHORT_QUERY_TEMPLATE.format(
            diabetes_exclusion_concepts=DIABETES_EXCLUSION_TABLE,
            diabetes_inclusion_concepts=DIABETES_INCLUSION_TABLE,
            age_lower_bound=self._age_lower_bound,
            date_lower_bound=self._date_lower_bound)
        return self.spark.sql(cohort_query)

    def extract_ehr_records_for_cohort(self, cohort: DataFrame):
        raise NotImplemented('extract_ehr_records_for_cohort not implemented '
                             'for TypeTwoDiabetesCohortBuilder')


def main(cohort_name, input_folder, output_folder, date_lower_bound, date_upper_bound,
         age_lower_bound, age_upper_bound, observation_window, prediction_window,
         index_date_match_window):
    cohort_builder = TypeTwoDiabetesCohortBuilder(cohort_name,
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
                                                  DEPENDENCY_LIST,
                                                  False)
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
