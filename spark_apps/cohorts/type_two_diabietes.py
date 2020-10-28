from spark_apps.cohorts.query_builder import QueryBuilder, QuerySpec, AncestorTableSpec

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
    c.visit_occurrence_id
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
"""

DIABETES_INCLUSION = [443238, 201820, 442793]
DIABETES_EXCLUSION = [40484648, 201254, 435216, 4058243]

DEPENDENCY_LIST = ['person', 'condition_occurrence', 'visit_occurrence']

DIABETES_INCLUSION_TABLE = 'diabetes_inclusion_concepts'
DIABETES_EXCLUSION_TABLE = 'diabetes_exclusion_concepts'

DEFAULT_COHORT_NAME = 'type_two_diabetes'


def query_builder():
    query = QuerySpec(table_name=DEFAULT_COHORT_NAME,
                      query_template=COHORT_QUERY_TEMPLATE,
                      parameters={'diabetes_exclusion_concepts': DIABETES_EXCLUSION_TABLE,
                                  'diabetes_inclusion_concepts': DIABETES_INCLUSION_TABLE})

    ancestor_table_specs = [AncestorTableSpec(table_name=DIABETES_INCLUSION_TABLE,
                                              ancestor_concept_ids=DIABETES_INCLUSION,
                                              is_standard=True),
                            AncestorTableSpec(table_name=DIABETES_EXCLUSION_TABLE,
                                              ancestor_concept_ids=DIABETES_EXCLUSION,
                                              is_standard=True)
                            ]
    return QueryBuilder(cohort_name=DEFAULT_COHORT_NAME,
                        dependency_list=DEPENDENCY_LIST,
                        query=query,
                        ancestor_table_specs=ancestor_table_specs)
