from spark_apps.cohorts.query_builder import QueryBuilder, QuerySpec, AncestorTableSpec

COHORT_QUERY_TEMPLATE = """
SELECT DISTINCT
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
    JOIN global_temp.visit_occurrence AS vo
        ON co.visit_occurrence_id = vo.visit_occurrence_id
    WHERE EXISTS (
        SELECT 1 
        FROM global_temp.{cad_concept_table} AS ie
        WHERE co.condition_concept_id = ie.concept_id
    )
) c
WHERE c.index_date >= '{date_lower_bound}'
"""

DEFAULT_COHORT_NAME = 'coronary_artery_disease'
DEPENDENCY_LIST = ['person', 'condition_occurrence', 'visit_occurrence']
CAD_INCLUSION_TABLE = 'CAD'
CAD_CONCEPTS = [317576]


def query_builder(spark_args):
    query = QuerySpec(
        table_name=DEFAULT_COHORT_NAME,
        query_template=COHORT_QUERY_TEMPLATE,
        parameters={
            'cad_concept_table': CAD_INCLUSION_TABLE,
            'date_lower_bound': spark_args.date_lower_bound
        }
    )

    ancestor_table_specs = [
        AncestorTableSpec(
            table_name=CAD_INCLUSION_TABLE,
            ancestor_concept_ids=CAD_CONCEPTS,
            is_standard=True
        )

    ]
    return QueryBuilder(
        cohort_name=DEFAULT_COHORT_NAME,
        dependency_list=DEPENDENCY_LIST,
        query=query,
        ancestor_table_specs=ancestor_table_specs
    )
