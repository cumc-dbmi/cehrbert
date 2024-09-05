from ..cohorts.query_builder import QueryBuilder, QuerySpec, AncestorTableSpec

COHORT_QUERY_TEMPLATE = """
SELECT
    co.person_id,
    FIRST(DATE(vo.visit_start_date)) OVER (PARTITION BY co.person_id 
        ORDER BY DATE(vo.visit_start_date), vo.visit_occurrence_id) AS index_date,
    FIRST(vo.visit_occurrence_id) OVER (PARTITION BY co.person_id 
        ORDER BY DATE(vo.visit_start_date), vo.visit_occurrence_id) AS visit_occurrence_id
FROM global_temp.condition_occurrence AS co 
JOIN global_temp.visit_occurrence AS vo
    ON co.visit_occurrence_id = vo.visit_occurrence_id
JOIN global_temp.{ischemic_stroke_concepts} AS c
    ON co.condition_concept_id = c.concept_id
"""

ISCHEMIC_STROKE_CONCEPT_ID = [443454]

DEPENDENCY_LIST = ['person', 'condition_occurrence', 'visit_occurrence']

DEFAULT_COHORT_NAME = 'ischemic_stroke'
ISCHEMIC_STROKE_CONCEPTS = 'ischemic_stroke_concepts'


def query_builder():
    query = QuerySpec(table_name=DEFAULT_COHORT_NAME,
                      query_template=COHORT_QUERY_TEMPLATE,
                      parameters={'ischemic_stroke_concepts': ISCHEMIC_STROKE_CONCEPTS})

    ancestor_table_specs = [AncestorTableSpec(table_name=ISCHEMIC_STROKE_CONCEPTS,
                                              ancestor_concept_ids=ISCHEMIC_STROKE_CONCEPT_ID,
                                              is_standard=True)]
    return QueryBuilder(cohort_name=DEFAULT_COHORT_NAME,
                        dependency_list=DEPENDENCY_LIST,
                        query=query,
                        ancestor_table_specs=ancestor_table_specs)
