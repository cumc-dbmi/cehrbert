from spark_apps.cohorts.query_builder import QueryBuilder, QuerySpec

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
        v.discharge_to_concept_id,
        ROW_NUMBER() OVER(PARTITION BY v.person_id ORDER BY DATE(v.visit_end_date) DESC) AS rn
    FROM global_temp.visit_occurrence AS v
    WHERE v.visit_concept_id IN (9201, 262) --inpatient, er-inpatient
        AND v.visit_end_date IS NOT NULL
        AND v.discharge_to_concept_id NOT IN (4216643, 44814650, 8717, 8970, 8971) --discharge to home or other facilities
) AS v
    WHERE v.rn = 1
"""

DEPENDENCY_LIST = ['person', 'visit_occurrence']
DEFAULT_COHORT_NAME = 'last_visit_discharge_home'


def query_builder():
    query = QuerySpec(table_name=DEFAULT_COHORT_NAME,
                      query_template=COHORT_QUERY,
                      parameters={})
    return QueryBuilder(cohort_name=DEFAULT_COHORT_NAME,
                        dependency_list=DEPENDENCY_LIST,
                        query=query)
