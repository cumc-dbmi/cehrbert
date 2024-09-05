from ..cohorts.query_builder import QueryBuilder, QuerySpec

VENTILATION_COHORT_QUERY = """
SELECT DISTINCT
    vent.person_id,
    vent.earliest_placement_instant AS index_date,
    CAST(NULL AS INT) AS visit_occurrence_id
FROM global_temp.vent AS vent
"""

DEFAULT_COHORT_NAME = 'ventilation'
DEPENDENCY_LIST = ['vent']


def query_builder():
    query = QuerySpec(table_name=DEFAULT_COHORT_NAME,
                      query_template=VENTILATION_COHORT_QUERY,
                      parameters={})

    return QueryBuilder(cohort_name=DEFAULT_COHORT_NAME,
                        dependency_list=DEPENDENCY_LIST,
                        query=query)
