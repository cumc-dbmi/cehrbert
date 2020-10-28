from spark_apps.cohorts.query_builder import QueryBuilder, QuerySpec

COHORT_QUERY = """
SELECT DISTINCT
    v.person_id,
    v.visit_occurrence_id,
    v.index_date
FROM 
(
    SELECT DISTINCT
        v.person_id,
        FIRST(v.visit_occurrence_id) OVER(PARTITION BY v.person_id 
            ORDER BY DATE(v.visit_start_date) DESC) AS visit_occurrence_id,
        FIRST(DATE(v.visit_start_date)) OVER(PARTITION BY v.person_id 
            ORDER BY DATE(v.visit_start_date) DESC) AS index_date,
        FIRST(v.discharge_to_concept_id) OVER(PARTITION BY v.person_id 
            ORDER BY DATE(v.visit_start_date) DESC) AS discharge_to_concept_id
    FROM global_temp.visit_occurrence AS v
    WHERE v.discharge_to_concept_id = 8536 --discharge to home
) AS v
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
