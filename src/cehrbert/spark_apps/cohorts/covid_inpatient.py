from ..cohorts.query_builder import QueryBuilder, QuerySpec

COVID_COHORT_QUERY = """
WITH covid_positive AS
(

    SELECT DISTINCT
        ROW_NUMBER() OVER(ORDER BY c.person_id, c.index_date) AS test_row_number,
        c.*
    FROM
    (
        SELECT DISTINCT
            m.person_id,
            COALESCE(vo.visit_start_datetime, m.measurement_datetime) AS index_date,
            vo.visit_occurrence_id,
            vo.visit_concept_id
        FROM global_temp.measurement AS m
        LEFT JOIN global_temp.visit_occurrence AS vo
            ON m.visit_occurrence_id = vo.visit_occurrence_id
        WHERE measurement_concept_id IN (723475,723479,706178,723473,723474,586515,706177,706163,706180,706181)
            AND value_source_value = 'Detected'

        UNION

        SELECT DISTINCT
            co.person_id,
            COALESCE(vo.visit_start_datetime, co.condition_start_datetime) AS index_date,
            vo.visit_occurrence_id,
            vo.visit_concept_id
        FROM global_temp.condition_occurrence AS co
        LEFT JOIN global_temp.visit_occurrence AS vo
            ON co.visit_occurrence_id = vo.visit_occurrence_id
        WHERE condition_concept_id = 37311061
    ) c
),

covid_test_with_no_visit AS
(
    SELECT DISTINCT
        c.test_row_number,
        c.person_id,
        FIRST_VALUE(vo.visit_start_datetime) OVER(PARTITION BY c.person_id ORDER BY vo.visit_start_datetime DESC) AS index_date,
        FIRST_VALUE(vo.visit_occurrence_id) OVER(PARTITION BY c.person_id ORDER BY vo.visit_start_datetime DESC) AS visit_occurrence_id,
        FIRST_VALUE(vo.visit_concept_id) OVER(PARTITION BY c.person_id ORDER BY vo.visit_start_datetime DESC) AS visit_concept_id
    FROM covid_positive AS c
    JOIN global_temp.visit_occurrence AS vo
        ON c.person_id = vo.person_id AND c.index_date BETWEEN DATE_ADD(vo.visit_start_date, -7) AND vo.visit_start_date
    WHERE c.visit_occurrence_id IS NULL
),

all_covid_tests AS
(
    SELECT DISTINCT
        c.person_id,
        COALESCE(c.index_date, cn.index_date) AS index_date,
        COALESCE(c.visit_occurrence_id, cn.visit_occurrence_id) AS visit_occurrence_id,
        COALESCE(c.visit_concept_id, cn.visit_concept_id) AS visit_concept_id
    FROM covid_positive AS c
    LEFT JOIN covid_test_with_no_visit AS cn
        ON c.test_row_number = cn.test_row_number
)

SELECT DISTINCT
    person_id,
    FIRST_VALUE(vo.index_date) OVER(PARTITION BY vo.person_id ORDER BY vo.index_date) AS index_date,
    FIRST_VALUE(vo.visit_occurrence_id) OVER(PARTITION BY vo.person_id ORDER BY vo.index_date) AS visit_occurrence_id
FROM
(
    SELECT
        co.*
    FROM all_covid_tests AS co
    WHERE visit_concept_id IN (262, 9203, 9201)
) vo
"""

DEFAULT_COHORT_NAME = "covid19"
DEPENDENCY_LIST = ["person", "visit_occurrence", "measurement", "condition_occurrence"]


def query_builder():
    query = QuerySpec(
        table_name=DEFAULT_COHORT_NAME, query_template=COVID_COHORT_QUERY, parameters={}
    )

    return QueryBuilder(
        cohort_name=DEFAULT_COHORT_NAME, dependency_list=DEPENDENCY_LIST, query=query
    )
