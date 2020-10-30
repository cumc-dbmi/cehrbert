from spark_apps.cohorts.query_builder import QueryBuilder, AncestorTableSpec, QuerySpec

HEART_FAILURE_CONCEPT = 316139
DIURETIC_CONCEPT_ID = [4186999]
PACEMAKER_CONCEPT_IDS = [4244395, 4051938, 2107005, 2313863, 4142917, 2106987, 2107003, 4232614,
                         2106988, 2313861, 2313855, 4204395, 42738824, 42742524, 2001980, 35622031,
                         42738822, 2107027, 4215909, 2211371, 4049403, 42738819, 42738820, 2313948,
                         2313860]

HEART_FAILURE_COHORT_QUERY = """
WITH hf_concepts AS (
    SELECT DISTINCT 
        descendant_concept_id AS concept_id 
    FROM global_temp.concept_ancestor AS ca
    WHERE ca.ancestor_concept_id = 316139
),
hf_conditions AS (
    SELECT
        *
    FROM global_temp.condition_occurrence AS co
    JOIN hf_concepts AS hc
        ON co.condition_concept_id = hc.concept_id
)

SELECT
    c.person_id,
    c.earliest_visit_start_date AS index_date,
    c.earliest_visit_occurrence_id AS visit_occurrence_id,
    COUNT(c.visit_occurrence_id) OVER(PARTITION BY c.person_id) AS num_of_diagnosis
FROM
(
    SELECT DISTINCT
        v.person_id,
        v.visit_occurrence_id,
        first(DATE(c.condition_start_date)) OVER (PARTITION BY v.person_id 
            ORDER BY DATE(c.condition_start_date)) AS earliest_condition_start_date,
        first(DATE(v.visit_start_date)) OVER (PARTITION BY v.person_id 
            ORDER BY DATE(v.visit_start_date)) AS earliest_visit_start_date,
        first(v.visit_occurrence_id) OVER (PARTITION BY v.person_id
            ORDER BY DATE(v.visit_start_date)) AS earliest_visit_occurrence_id
    FROM global_temp.visit_occurrence AS v
    JOIN hf_conditions AS c
        ON v.visit_occurrence_id = c.visit_occurrence_id
) c
WHERE c.earliest_visit_start_date <= c.earliest_condition_start_date
"""

EXCLUDE_PRIOR_DIURETICS_TEMPLATE = """
WITH diuretics_user AS (
    SELECT DISTINCT
        de.person_id,
        FIRST(DATE(drug_exposure_start_date)) AS drug_exposure_start_date
    FROM global_temp.drug_exposure AS de 
    JOIN global_temp.{diuretics_concepts} AS dc
        ON de.drug_concept_id = dc.concept_id
    GROUP BY de.person_id
)
SELECT DISTINCT
    c.*
FROM global_temp.{cohort_table_name} AS c 
LEFT JOIN diuretics_user AS du
    ON c.person_id = du.person_id AND c.index_date > du.drug_exposure_start_date
WHERE du.person_id IS NULL
"""

EXCLUDE_PRIOR_PACEMAKERS_TEMPLATE = """
WITH pacemaker_user AS (
    SELECT DISTINCT
        de.person_id,
        FIRST(DATE(de.procedure_date)) AS procedure_date
    FROM global_temp.procedure_occurrence AS de 
    WHERE de.procedure_concept_id IN ({procedure_concept_ids})
    GROUP BY de.person_id
)
SELECT DISTINCT
    c.*
FROM global_temp.{cohort_table_name} AS c 
LEFT JOIN pacemaker_user AS pu
    ON c.person_id = pu.person_id AND c.index_date > pu.procedure_date
WHERE pu.person_id IS NULL
"""

ROLL_UP_DIURETICS_TO_INGREDIENT_TEMPLATE = """
SELECT DISTINCT
    c.*
FROM global_temp.diuretics_ancestor_table AS a
JOIN global_temp.concept_relationship AS cr
    ON a.descendant_concept_id = cr.concept_id_1 AND cr.relationship_id = 'Maps to'
JOIN global_temp.concept_ancestor AS ca
    ON cr.concept_id_2 = ca.descendant_concept_id
JOIN global_temp.concept AS c
    ON ca.ancestor_concept_id = c.concept_id
WHERE c.concept_class_id = 'Ingredient'
"""

DEPENDENCY_LIST = ['person', 'condition_occurrence', 'visit_occurrence', 'drug_exposure',
                   'procedure_occurrence']

DIURETICS_INGREDIENT_CONCEPTS = 'diuretics_concepts'
DIURETICS_ANCESTOR_TABLE = 'diuretics_ancestor_table'
DEFAULT_COHORT_NAME = 'heart_failure'


def query_builder():
    dependency_queries = [QuerySpec(table_name=DIURETICS_INGREDIENT_CONCEPTS,
                                    query_template=ROLL_UP_DIURETICS_TO_INGREDIENT_TEMPLATE,
                                    parameters={})]
    query = QuerySpec(table_name=DEFAULT_COHORT_NAME,
                      query_template=HEART_FAILURE_COHORT_QUERY,
                      parameters={})
    post_queries = [QuerySpec(table_name=DEFAULT_COHORT_NAME,
                              query_template=EXCLUDE_PRIOR_DIURETICS_TEMPLATE,
                              parameters={'cohort_table_name': DEFAULT_COHORT_NAME,
                                          'diuretics_concepts': DIURETICS_INGREDIENT_CONCEPTS}),
                    QuerySpec(table_name=DEFAULT_COHORT_NAME,
                              query_template=EXCLUDE_PRIOR_PACEMAKERS_TEMPLATE,
                              parameters={'cohort_table_name': DEFAULT_COHORT_NAME,
                                          'procedure_concept_ids': ','.join(
                                              [str(c) for c in PACEMAKER_CONCEPT_IDS])})
                    ]

    ancestor_table_specs = [AncestorTableSpec(table_name=DIURETICS_ANCESTOR_TABLE,
                                              ancestor_concept_ids=DIURETIC_CONCEPT_ID,
                                              is_standard=False)]

    return QueryBuilder(cohort_name=DEFAULT_COHORT_NAME,
                        query=query,
                        dependency_list=DEPENDENCY_LIST,
                        dependency_queries=dependency_queries,
                        post_queries=post_queries,
                        ancestor_table_specs=ancestor_table_specs)
