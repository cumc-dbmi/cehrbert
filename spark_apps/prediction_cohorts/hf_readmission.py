from spark_apps.parameters import create_spark_args
from spark_apps.cohorts.spark_app_base import create_prediction_cohort
from spark_apps.cohorts.query_builder import QueryBuilder, AncestorTableSpec, QuerySpec

HEART_FAILURE_HOSPITALIZATION_QUERY = """
WITH hf_concepts AS (
    SELECT DISTINCT 
        descendant_concept_id AS concept_id 
    FROM global_temp.concept_ancestor AS ca
    WHERE ca.ancestor_concept_id = 316139
),
hf_conditions AS (
    SELECT
        *
    FROM global_temp.condition_occurrence AS co
    JOIN hf_concepts AS hc
        ON co.condition_concept_id = hc.concept_id
)

SELECT
    c.person_id,
    c.earliest_visit_start_date AS index_date,
    c.earliest_visit_occurrence_id AS visit_occurrence_id
FROM
(
    SELECT DISTINCT
        v.person_id,
        v.visit_occurrence_id,
        first(DATE(c.condition_start_date)) OVER (PARTITION BY v.person_id 
            ORDER BY DATE(c.condition_start_date)) AS earliest_condition_start_date,
        first(DATE(v.visit_start_date)) OVER (PARTITION BY v.person_id 
            ORDER BY DATE(v.visit_start_date)) AS earliest_visit_start_date,
        first(v.visit_occurrence_id) OVER (PARTITION BY v.person_id
            ORDER BY DATE(v.visit_start_date)) AS earliest_visit_occurrence_id
    FROM global_temp.visit_occurrence AS v
    JOIN global_temp.hf_conditions AS c
        ON v.visit_occurrence_id = c.visit_occurrence_id
    WHERE v.visit_concept_id IN (9201, 262) --inpatient, er-inpatient
) c
WHERE c.earliest_visit_start_date <= c.earliest_condition_start_date
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
    hospitalization = QueryBuilder(cohort_name=HOSPITALIZATION_QUERY,
                                   dependency_list=DEPENDENCY_LIST,
                                   query=hospitalization_query)

    create_prediction_cohort(spark_args,
                             hf_inpatient,
                             hospitalization,
                             DOMAIN_TABLE_LIST)


if __name__ == '__main__':
    main(create_spark_args())
