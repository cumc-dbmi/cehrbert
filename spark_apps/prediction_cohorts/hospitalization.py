from spark_apps.parameters import create_spark_args
from spark_apps.cohorts.spark_app_base import create_prediction_cohort
from spark_apps.cohorts.query_builder import QueryBuilder, QuerySpec

HOSPITALIZATION_OUTCOME_QUERY = """
SELECT DISTINCT
    v.person_id,
    visit_start_date AS index_date,
    visit_occurrence_id
FROM global_temp.visit_occurrence AS v
WHERE v.visit_concept_id IN (9201, 262)
"""

HOSPITALIZATION_TARGET_QUERY = """
SELECT
    v.person_id,
    v.observation_period_start_date AS index_date,
    CAST(null AS INT) AS visit_occurrence_id
FROM
(
    SELECT
        v.person_id,
        v.observation_period_start_date,
        SUM(CASE WHEN v.visit_concept_id IN (9201, 262) THEN 1 ELSE 0 END) AS num_of_hospitalizations
    FROM
    (
        SELECT DISTINCT
            op.person_id,
            op.observation_period_start_date,
            v2.visit_concept_id,
            v2.visit_start_date,
            v2.visit_occurrence_id
        FROM global_temp.observation_period AS op
        LEFT JOIN global_temp.visit_occurrence AS v2
            ON op.person_id = v2.person_id
                AND DATEDIFF(v2.visit_start_date, op.observation_period_start_date) <= {total_window}
    ) v
    GROUP BY v.observation_period_start_date, v.person_id
) v
WHERE v.num_of_hospitalizations = 0
"""

HOSPITALIZATION_TARGET_COHORT = 'hospitalization_target'
HOSPITALIZATION_OUTCOME_COHORT = 'hopitalization_outcome'
DEPENDENCY_LIST = ['person', 'condition_occurrence', 'visit_occurrence']
DOMAIN_TABLE_LIST = ['condition_occurrence', 'drug_exposure', 'procedure_occurrence']


def main(spark_args):
    total_window = spark_args.observation_window + spark_args.hold_off_window
    hospitalization_target_query = QuerySpec(table_name=HOSPITALIZATION_TARGET_COHORT,
                                             query_template=HOSPITALIZATION_TARGET_QUERY,
                                             parameters={'total_window': total_window})
    hf_inpatient = QueryBuilder(cohort_name=HOSPITALIZATION_TARGET_COHORT,
                                dependency_list=DEPENDENCY_LIST,
                                query=hospitalization_target_query)

    hospitalization_outcome_query = QuerySpec(table_name=HOSPITALIZATION_OUTCOME_COHORT,
                                              query_template=HOSPITALIZATION_OUTCOME_QUERY,
                                              parameters={})
    hospitalization = QueryBuilder(cohort_name=HOSPITALIZATION_OUTCOME_COHORT,
                                   dependency_list=DEPENDENCY_LIST,
                                   query=hospitalization_outcome_query)

    create_prediction_cohort(spark_args,
                             hf_inpatient,
                             hospitalization,
                             DOMAIN_TABLE_LIST)


if __name__ == '__main__':
    main(create_spark_args())
