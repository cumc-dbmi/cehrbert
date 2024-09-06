from ..cohorts.query_builder import QueryBuilder, QuerySpec, AncestorTableSpec

COHORT_QUERY_TEMPLATE = """
WITH prior_graft_stent AS (
    SELECT
        po.person_id,
        po.procedure_date
    FROM global_temp.procedure_occurrence AS po
    WHERE EXISTS (
        SELECT 1
        FROM global_temp.{graft_stent_table} AS gs
        WHERE po.procedure_concept_id = gs.concept_id
    )
)
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
WHERE NOT EXISTS (
    -- The patients who had a graft or stent procedures before the index date 
    -- need to be removed from the cohort
    SELECT 1
    FROM prior_graft_stent AS exclusion
    WHERE exclusion.person_id = c.person_id 
        AND c.index_date > exclusion.procedure_date
) AND c.index_date >= '{date_lower_bound}' 
"""

DEFAULT_COHORT_NAME = "coronary_artery_disease"
DEPENDENCY_LIST = [
    "person",
    "condition_occurrence",
    "procedure_occurrence",
    "visit_occurrence",
]
CAD_INCLUSION_TABLE = "CAD"
CAD_CONCEPTS = [317576]

PRIOR_PROCEDURE_TABLE = "graft_stent"
PRIOR_PROCEDURES = [4296227, 42537730, 762043, 44782770, 42537729]


def query_builder(spark_args):
    query = QuerySpec(
        table_name=DEFAULT_COHORT_NAME,
        query_template=COHORT_QUERY_TEMPLATE,
        parameters={
            "cad_concept_table": CAD_INCLUSION_TABLE,
            "graft_stent_table": PRIOR_PROCEDURE_TABLE,
            "date_lower_bound": spark_args.date_lower_bound,
        },
    )

    ancestor_table_specs = [
        AncestorTableSpec(
            table_name=CAD_INCLUSION_TABLE,
            ancestor_concept_ids=CAD_CONCEPTS,
            is_standard=True,
        ),
        AncestorTableSpec(
            table_name=PRIOR_PROCEDURE_TABLE,
            ancestor_concept_ids=PRIOR_PROCEDURES,
            is_standard=True,
        ),
    ]
    return QueryBuilder(
        cohort_name=DEFAULT_COHORT_NAME,
        dependency_list=DEPENDENCY_LIST,
        query=query,
        ancestor_table_specs=ancestor_table_specs,
    )
