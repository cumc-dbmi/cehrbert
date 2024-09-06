from ..cohorts.query_builder import AncestorTableSpec, QueryBuilder, QuerySpec

COHORT_QUERY_TEMPLATE = """
WITH person_ids_to_include_drug AS
(
    SELECT DISTINCT
        d.person_id
    FROM global_temp.drug_exposure AS d
    JOIN global_temp.{drug_inclusion_concepts} AS e
        ON d.drug_concept_id = e.concept_id
),
person_ids_to_exclude_observation AS
(

    SELECT DISTINCT
        o.person_id,
        o.observation_date
    FROM global_temp.observation AS o
    JOIN global_temp.{observation_exclusion_concepts} AS oec
        ON o.observation_concept_id = oec.concept_id
)
SELECT
    distinct
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
) c
JOIN person_ids_to_include_drug AS d
    ON c.person_id = d.person_id
LEFT JOIN person_ids_to_exclude_observation AS eo
    ON c.person_id = eo.person_id AND c.index_date > eo.observation_date
WHERE eo.person_id IS NULL AND c.index_date >= '{date_lower_bound}'
"""

DIABETES_INCLUSION = [443238, 201820, 442793, 4016045]
DIABETES_EXCLUSION = [
    40484648,
    201254,
    435216,
    4058243,
    30968,
    438476,
    195771,
    193323,
    4019513,
    40484649,
]
DRUG_INCLUSION = [
    1503297,
    1594973,
    1597756,
    1559684,
    1560171,
    1502855,
    1502809,
    1525215,
    1547504,
    1580747,
    40166035,
    43013884,
    40239216,
    1516766,
    1502826,
    1510202,
    1529331,
    35605670,
    35602717,
    1516976,
    1502905,
    46221581,
    1550023,
    35198096,
    42899447,
    1544838,
    1567198,
    35884381,
    1531601,
    1588986,
    1513876,
    19013951,
    1590165,
    1596977,
    1586346,
    19090204,
    1513843,
    1513849,
    1562586,
    19090226,
    19090221,
    1586369,
    19090244,
    19090229,
    19090247,
    19090249,
    19090180,
    19013926,
    19091621,
    19090187,
]
OBSERVATION_EXCLUSION = [40769338, 43021173, 42539022, 46270562]
DEPENDENCY_LIST = [
    "person",
    "condition_occurrence",
    "visit_occurrence",
    "drug_exposure",
    "observation",
]

DIABETES_INCLUSION_TABLE = "diabetes_inclusion_concepts"
DIABETES_EXCLUSION_TABLE = "diabetes_exclusion_concepts"
DRUG_INCLUSION_TABLE = "drug_inclusion_concepts"
OBSERVATION_EXCLUSION_TABLE = "observation_exclusion_concepts"

DEFAULT_COHORT_NAME = "type_two_diabetes"


def query_builder(spark_args):
    query = QuerySpec(
        table_name=DEFAULT_COHORT_NAME,
        query_template=COHORT_QUERY_TEMPLATE,
        parameters={
            "diabetes_exclusion_concepts": DIABETES_EXCLUSION_TABLE,
            "diabetes_inclusion_concepts": DIABETES_INCLUSION_TABLE,
            "drug_inclusion_concepts": DRUG_INCLUSION_TABLE,
            "observation_exclusion_concepts": OBSERVATION_EXCLUSION_TABLE,
            "date_lower_bound": spark_args.date_lower_bound,
        },
    )

    ancestor_table_specs = [
        AncestorTableSpec(
            table_name=DIABETES_INCLUSION_TABLE,
            ancestor_concept_ids=DIABETES_INCLUSION,
            is_standard=True,
        ),
        AncestorTableSpec(
            table_name=DIABETES_EXCLUSION_TABLE,
            ancestor_concept_ids=DIABETES_EXCLUSION,
            is_standard=True,
        ),
        AncestorTableSpec(
            table_name=OBSERVATION_EXCLUSION_TABLE,
            ancestor_concept_ids=OBSERVATION_EXCLUSION,
            is_standard=True,
        ),
        AncestorTableSpec(
            table_name=DRUG_INCLUSION_TABLE,
            ancestor_concept_ids=DRUG_INCLUSION,
            is_standard=True,
        ),
    ]
    return QueryBuilder(
        cohort_name=DEFAULT_COHORT_NAME,
        dependency_list=DEPENDENCY_LIST,
        query=query,
        ancestor_table_specs=ancestor_table_specs,
    )
