from ..cohorts.query_builder import QueryBuilder, AncestorTableSpec, QuerySpec, \
    create_cohort_entry_query_spec, create_negative_query_spec

# 1. Incidens of Heart Failure
HEART_FAILURE_CONCEPT = [316139]

# 2. At least one new or worsening symptoms due to HF
WORSEN_HF_DIAGNOSIS_CONCEPT = [312437, 4263848, 46272935, 4223659, 315361]

# 3. At least TWO physical examination findings OR one physical examination finding and at least
# ONE laboratory criterion
PHYSICAL_EXAM_CONCEPT = [433595, 200528, 4117930, 4329988, 4289004, 4285133]
## Lab result concept
# https://labtestsonline.org/tests/bnp-and-nt-probnp
BNP_CONCEPT = [4307029, 3031569, 3011960,
               3052295]  # High B-type Natriuretic Peptide (BNP) > 500 pg/mL
NT_PRO_BNP_CONCEPT = [3029187, 42529224, 3029435, 42529225]
PWP_CONCEPT = [1002721, 4040920,
               21490776]  # Pulmonary artery wedge pressure >= 18 no patient in cumc
CVP_CONCEPT = [21490675, 4323687, 3000333,
               1003995]  # Central venous pressure >= 12 no patient in cumc
CI_CONCEPT = 21490712  # Cardiac index < 2.2 no patient in cumc

# 4. At least ONE of the treatments specifically for HF
DRUG_CONCEPT = [956874, 942350, 987406, 932745, 1309799, 970250, 992590, 907013, 1942960]
MECHANICAL_CIRCULATORY_SUPPORT_CONCEPT = [45888564, 4052536, 4337306, 2107514, 45889695, 2107500,
                                          45887675, 43527920, 2107501, 45890116, 40756954, 4338594,
                                          43527923, 40757060, 2100812]
DIALYSIS_CONCEPT = [4032243, 45889365]
ARTIFICIAL_HEART_ASSOCIATED_PROCEDURE_CONCEPT = [4144390, 4150347, 4281764, 725038, 725037, 2100816,
                                                 2100822, 725039, 2100828, 4337306, 4140024,
                                                 4146121, 4060257, 4309033, 4222272, 4243758,
                                                 4241906, 4080968, 4224193, 4052537, 4050864]

DIURETIC_CONCEPT_ID = [4186999]

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

HEART_FAILURE_ENTRY_COHORT = """
WITH hf_conditions AS (
SELECT
    *
FROM global_temp.condition_occurrence AS co
JOIN global_temp.{hf_concept} AS hf
ON co.condition_concept_id = hf.concept_id
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

HEART_FAILURE_INTERMEDIATE_COHORT_QUERY = """
WITH hf_conditions AS (
    SELECT
        *
    FROM global_temp.condition_occurrence AS co
    JOIN global_temp.{hf_concept} AS hf
    ON co.condition_concept_id = hf.concept_id
),

worsen_hf_diagnosis AS (
    SELECT DISTINCT person_id, visit_occurrence_id 
    FROM global_temp.condition_occurrence AS co
    JOIN global_temp.{worsen_hf_dx_concepts} AS w_hf
    ON co.condition_concept_id = w_hf.concept_id
),

phy_exam_cohort AS (
    SELECT DISTINCT person_id, visit_occurrence_id
    FROM global_temp.condition_occurrence AS co
    JOIN global_temp.{phy_exam_concepts} AS phy
    ON co.condition_concept_id = phy.concept_id
),

bnp_cohort AS (
    SELECT DISTINCT person_id, visit_occurrence_id
    FROM global_temp.measurement AS m
    JOIN global_temp.{bnp_concepts} AS bnp
    ON m.measurement_concept_id = bnp.concept_id
    AND m.value_source_value > 500
    UNION ALL 
    SELECT DISTINCT person_id, visit_occurrence_id
    FROM global_temp.measurement AS m
    JOIN global_temp.{nt_pro_bnp_concepts} AS nt_bnp
    ON m.measurement_concept_id = nt_bnp.concept_id
    AND m.value_source_value > 2000
),

drug_concepts AS (
    SELECT DISTINCT 
        *
    FROM
    (
        SELECT *  
        FROM global_temp.{drug_concepts} 
        
        UNION 
        
        SELECT  *
        FROM global_temp.diuretics_concepts
    ) d
),

drug_cohort AS (
    SELECT DISTINCT person_id, visit_occurrence_id
    FROM global_temp.drug_exposure AS d
    JOIN drug_concepts AS dc
    ON d.drug_concept_id = dc.concept_id
),

mechanical_support_cohort AS (
    SELECT DISTINCT person_id, visit_occurrence_id
    FROM global_temp.procedure_occurrence AS p
    JOIN global_temp.{mechanical_support_concepts} AS msc
    ON p.procedure_concept_id = msc.concept_id
),

dialysis_cohort AS (
    SELECT DISTINCT person_id, visit_occurrence_id
    FROM global_temp.procedure_occurrence AS p
    JOIN global_temp.{dialysis_concepts} AS dc
    ON p.procedure_concept_id = dc.concept_id
),

artificial_heart_cohort AS (
    SELECT DISTINCT person_id, visit_occurrence_id
    FROM global_temp.procedure_occurrence AS p
    JOIN global_temp.{artificial_heart_concepts} AS ahc
    ON p.procedure_concept_id = ahc.concept_id
),

treatment_cohort AS (
--    SELECT * FROM drug_cohort
--    UNION ALL
    SELECT * FROM mechanical_support_cohort
    UNION ALL
    SELECT * FROM dialysis_cohort
    UNION ALL
    SELECT * FROM artificial_heart_cohort
),

entry_cohort AS (
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
)

SELECT
    c.*,
    CAST(COALESCE(bnp.person_id, tc.person_id, dc.person_id) IS NOT NULL AS INT) AS inclusion
FROM entry_cohort AS c
LEFT JOIN (
SELECT DISTINCT person_id FROM bnp_cohort
) AS bnp
    ON c.person_id = bnp.person_id
LEFT JOIN (
    SELECT DISTINCT 
        person_id
    FROM treatment_cohort
) AS tc
    ON c.person_id = tc.person_id
LEFT JOIN (
    SELECT DISTINCT 
        hf.person_id
    FROM hf_conditions hf
    JOIN drug_cohort dc
    ON hf.visit_occurrence_id = dc.visit_occurrence_id
) AS dc
    ON c.person_id = dc.person_id
"""

HEART_FAILURE_COHORT_QUERY = """
SELECT
    person_id,
    index_date,
    visit_occurrence_id
FROM global_temp.{intermediate_heart_failure}
WHERE inclusion = {inclusion}
"""

DEPENDENCY_LIST = ['person', 'condition_occurrence', 'visit_occurrence', 'drug_exposure',
                   'measurement',
                   'procedure_occurrence']
HEART_FAILURE_CONCEPT_TABLE = 'hf_concept'
WORSEN_HF_DX_CONCEPT_TABLE = 'worsen_hf_dx_concepts'
PHYSICAL_EXAM_COHORT_TABLE = 'phy_exam_concepts'
BNP_CONCEPT_TABLE = 'bnp_concepts'
NT_PRO_BNP_CONCEPT_TABLE = 'nt_pro_bnp_concepts'
DRUG_CONCEPT_TABLE = 'drug_concepts'
MECHANICAL_SUPPORT_CONCEPT_TABLE = 'mechanical_support_concepts'
DIALYSIS_CONCEPT_TABLE = 'dialysis_concepts'
ARTIFICIAL_HEART_CONCEPT_TABLE = 'artificial_heart_concepts'

DIURETICS_ANCESTOR_TABLE = 'diuretics_ancestor_table'
DIURETICS_INGREDIENT_CONCEPTS = 'diuretics_concepts'

INTERMEDIATE_COHORT_NAME = 'intermediate_heart_failure'
DEFAULT_COHORT_NAME = 'heart_failure'
NEGATIVE_COHORT_NAME = 'negative_heart_failure'


def query_builder():
    query = QuerySpec(table_name=DEFAULT_COHORT_NAME,
                      query_template=HEART_FAILURE_COHORT_QUERY,
                      parameters={'intermediate_heart_failure': INTERMEDIATE_COHORT_NAME,
                                  'inclusion': 1})

    ancestor_table_specs = [AncestorTableSpec(table_name=HEART_FAILURE_CONCEPT_TABLE,
                                              ancestor_concept_ids=HEART_FAILURE_CONCEPT,
                                              is_standard=True),
                            AncestorTableSpec(table_name=WORSEN_HF_DX_CONCEPT_TABLE,
                                              ancestor_concept_ids=WORSEN_HF_DIAGNOSIS_CONCEPT,
                                              is_standard=True),
                            AncestorTableSpec(table_name=PHYSICAL_EXAM_COHORT_TABLE,
                                              ancestor_concept_ids=PHYSICAL_EXAM_CONCEPT,
                                              is_standard=True),
                            AncestorTableSpec(table_name=BNP_CONCEPT_TABLE,
                                              ancestor_concept_ids=BNP_CONCEPT,
                                              is_standard=True),
                            AncestorTableSpec(table_name=NT_PRO_BNP_CONCEPT_TABLE,
                                              ancestor_concept_ids=NT_PRO_BNP_CONCEPT,
                                              is_standard=True),
                            AncestorTableSpec(table_name=DRUG_CONCEPT_TABLE,
                                              ancestor_concept_ids=DRUG_CONCEPT,
                                              is_standard=True),
                            AncestorTableSpec(table_name=MECHANICAL_SUPPORT_CONCEPT_TABLE,
                                              ancestor_concept_ids=MECHANICAL_CIRCULATORY_SUPPORT_CONCEPT,
                                              is_standard=True),
                            AncestorTableSpec(table_name=DIALYSIS_CONCEPT_TABLE,
                                              ancestor_concept_ids=DIALYSIS_CONCEPT,
                                              is_standard=True),
                            AncestorTableSpec(table_name=ARTIFICIAL_HEART_CONCEPT_TABLE,
                                              ancestor_concept_ids=ARTIFICIAL_HEART_ASSOCIATED_PROCEDURE_CONCEPT,
                                              is_standard=True),
                            AncestorTableSpec(table_name=DIURETICS_ANCESTOR_TABLE,
                                              ancestor_concept_ids=DIURETIC_CONCEPT_ID,
                                              is_standard=False)
                            ]

    dependency_queries = [QuerySpec(table_name=DIURETICS_INGREDIENT_CONCEPTS,
                                    query_template=ROLL_UP_DIURETICS_TO_INGREDIENT_TEMPLATE,
                                    parameters={}),
                          QuerySpec(table_name=INTERMEDIATE_COHORT_NAME,
                                    query_template=HEART_FAILURE_INTERMEDIATE_COHORT_QUERY,
                                    parameters={'hf_concept': HEART_FAILURE_CONCEPT_TABLE,
                                                'worsen_hf_dx_concepts': WORSEN_HF_DX_CONCEPT_TABLE,
                                                'phy_exam_concepts': PHYSICAL_EXAM_COHORT_TABLE,
                                                'bnp_concepts': BNP_CONCEPT_TABLE,
                                                'nt_pro_bnp_concepts': NT_PRO_BNP_CONCEPT_TABLE,
                                                'drug_concepts': DRUG_CONCEPT_TABLE,
                                                'mechanical_support_concepts': MECHANICAL_SUPPORT_CONCEPT_TABLE,
                                                'dialysis_concepts': DIALYSIS_CONCEPT_TABLE,
                                                'artificial_heart_concepts': ARTIFICIAL_HEART_CONCEPT_TABLE
                                                })]

    entry_cohort_query = create_cohort_entry_query_spec(
        entry_query_template=HEART_FAILURE_ENTRY_COHORT,
        parameters={'hf_concept': HEART_FAILURE_CONCEPT_TABLE})

    negative_query = create_negative_query_spec(
        entry_query_template=HEART_FAILURE_COHORT_QUERY,
        parameters={'intermediate_heart_failure': INTERMEDIATE_COHORT_NAME,
                    'inclusion': 0})

    return QueryBuilder(cohort_name=DEFAULT_COHORT_NAME,
                        query=query,
                        negative_query=negative_query,
                        entry_cohort_query=entry_cohort_query,
                        dependency_list=DEPENDENCY_LIST,
                        dependency_queries=dependency_queries,
                        post_queries=[],
                        ancestor_table_specs=ancestor_table_specs)
