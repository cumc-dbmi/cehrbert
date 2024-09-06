from ..cohorts.query_builder import QueryBuilder, QuerySpec

COHORT_QUERY = """
SELECT DISTINCT
    v.person_id,
    v.visit_occurrence_id,
    v.index_date
FROM
(
    SELECT
        v.person_id,
        v.visit_occurrence_id,
        v.visit_end_date AS index_date,
        v.discharged_to_concept_id,
        ROW_NUMBER() OVER(PARTITION BY v.person_id ORDER BY DATE(v.visit_end_date) DESC) AS rn
    FROM global_temp.visit_occurrence AS v
    WHERE v.visit_concept_id IN (9201, 262) --inpatient, er-inpatient
        AND v.visit_end_date IS NOT NULL
        AND v.discharged_to_concept_id = 8536 --discharge to home
) AS v
    WHERE v.rn = 1 AND v.index_date >= '{date_lower_bound}'
"""

DEPENDENCY_LIST = ["person", "visit_occurrence"]
DEFAULT_COHORT_NAME = "last_visit_discharge_home"


def query_builder(spark_args):
    query = QuerySpec(
        table_name=DEFAULT_COHORT_NAME,
        query_template=COHORT_QUERY,
        parameters={"date_lower_bound": spark_args.date_lower_bound},
    )
    return QueryBuilder(
        cohort_name=DEFAULT_COHORT_NAME, dependency_list=DEPENDENCY_LIST, query=query
    )
