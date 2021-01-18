from spark_apps.cohorts.query_builder import QueryBuilder, QuerySpec

COVID_COHORT_QUERY = """
SELECT DISTINCT
    c.person_id,
    DATE_ADD(FIRST(index_date) OVER (PARTITION BY person_id ORDER BY index_date, visit_occurrence_id), 1) AS index_date,
    FIRST(visit_occurrence_id) OVER (PARTITION BY person_id ORDER BY index_date, visit_occurrence_id) AS visit_occurrence_id
FROM
(
    SELECT DISTINCT
        m.person_id,
        FIRST(visit_start_date) OVER (PARTITION BY v.person_id ORDER BY visit_start_date, v.visit_occurrence_id) AS index_date,
        FIRST(v.visit_occurrence_id) OVER (PARTITION BY v.person_id ORDER BY visit_start_date, v.visit_occurrence_id) AS visit_occurrence_id
    FROM global_temp.measurement AS m
    JOIN global_temp.visit_occurrence AS v
        ON m.visit_occurrence_id = v.visit_occurrence_id
    JOIN global_temp.concept AS c
        ON m.value_as_concept_id = c.concept_id
    WHERE m.measurement_concept_id IN (723475,723479,706178,723473,723474,586515,706177,706163,706180,706181)
        AND c.concept_name IN ('Detected', 'Positve')

    UNION

    SELECT 
        co.person_id,
        FIRST(visit_start_date) OVER (PARTITION BY v.person_id ORDER BY visit_start_date, v.visit_occurrence_id) AS index_date,
        FIRST(v.visit_occurrence_id) OVER (PARTITION BY v.person_id ORDER BY visit_start_date, v.visit_occurrence_id) AS visit_occurrence_id
    FROM global_temp.condition_occurrence AS co
    JOIN global_temp.visit_occurrence AS v
        ON co.visit_occurrence_id = v.visit_occurrence_id
    WHERE co.condition_concept_id = 37311061
) c
"""

DEFAULT_COHORT_NAME = 'covid19'
DEPENDENCY_LIST = ['person', 'visit_occurrence', 'measurement', 'condition_occurrence']


def query_builder():
    query = QuerySpec(table_name=DEFAULT_COHORT_NAME,
                      query_template=COVID_COHORT_QUERY,
                      parameters={})

    return QueryBuilder(cohort_name=DEFAULT_COHORT_NAME,
                        dependency_list=DEPENDENCY_LIST,
                        query=query)
