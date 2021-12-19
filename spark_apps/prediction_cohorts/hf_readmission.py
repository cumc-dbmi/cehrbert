from spark_apps.spark_parse_args import create_spark_args
from spark_apps.cohorts.spark_app_base import create_prediction_cohort
from spark_apps.cohorts.query_builder import QueryBuilder, QuerySpec

HEART_FAILURE_HOSPITALIZATION_QUERY = """
WITH hf_concepts AS (
    SELECT DISTINCT 
        descendant_concept_id AS concept_id 
    FROM global_temp.concept_ancestor AS ca
    WHERE ca.ancestor_concept_id = 316139
)

SELECT DISTINCT
    v.person_id,
    v.visit_occurrence_id,
    v.visit_end_date AS index_date
FROM global_temp.visit_occurrence AS v
JOIN global_temp.condition_occurrence AS co
    ON v.visit_occurrence_id = co.visit_occurrence_id
JOIN hf_concepts AS hf
    ON co.condition_concept_id = hf.concept_id
WHERE v.visit_concept_id IN (9201, 262) --inpatient, er-inpatient
    AND v.discharge_to_concept_id NOT IN (4216643, 44814650, 8717, 8970, 8971) -- TBD
    --AND v.discharge_to_concept_id IN (8536, 8863, 4161979) -- Home, Skilled Nursing Facility, and Patient discharged alive
    AND v.visit_start_date <= co.condition_start_date
"""

HOSPITALIZATION_QUERY = """
SELECT DISTINCT
    v.person_id,
    v.visit_occurrence_id,
    v.visit_start_date AS index_date
FROM global_temp.visit_occurrence AS v
WHERE v.visit_concept_id IN (9201, 262) --inpatient, er-inpatient
"""

HF_HOSPITALIZATION_COHORT = 'hf_hospitalization'
HOSPITALIZATION_COHORT = 'hospitalization'
DEPENDENCY_LIST = ['person', 'condition_occurrence', 'visit_occurrence']
DOMAIN_TABLE_LIST = ['condition_occurrence', 'drug_exposure', 'procedure_occurrence']


def main(spark_args):
    hf_inpatient_query = QuerySpec(table_name=HF_HOSPITALIZATION_COHORT,
                                   query_template=HEART_FAILURE_HOSPITALIZATION_QUERY,
                                   parameters={})
    hf_inpatient = QueryBuilder(cohort_name=HF_HOSPITALIZATION_COHORT,
                                dependency_list=DEPENDENCY_LIST,
                                query=hf_inpatient_query)

    hospitalization_query = QuerySpec(table_name=HOSPITALIZATION_COHORT,
                                      query_template=HOSPITALIZATION_QUERY,
                                      parameters={})
    hospitalization = QueryBuilder(cohort_name=HOSPITALIZATION_COHORT,
                                   dependency_list=DEPENDENCY_LIST,
                                   query=hospitalization_query)

    create_prediction_cohort(spark_args,
                             hf_inpatient,
                             hospitalization,
                             DOMAIN_TABLE_LIST)


if __name__ == '__main__':
    main(create_spark_args())
