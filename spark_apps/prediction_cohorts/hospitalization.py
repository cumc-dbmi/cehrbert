from spark_apps.cohorts.spark_app_base import *
from spark_apps.parameters import create_spark_args

OUTCOME_COHORT_QUERY_TEMPLATE = """
SELECT DISTINCT
    v.person_id,
    FIRST(visit_start_date) OVER (PARTITION BY person_id ORDER BY visit_start_date, visit_occurrence_id) AS index_date,
    FIRST(visit_occurrence_id) OVER (PARTITION BY person_id ORDER BY visit_start_date, visit_occurrence_id) AS visit_occurrence_id
FROM global_temp.visit_occurrence AS v
WHERE v.visit_concept_id IN {visit_concept_ids} 
"""

TARGET_COHORT_QUERY_TEMPLATE = """
SELECT
    v.person_id,
    v.observation_period_start_date AS index_date,
    CAST(null AS INT) AS visit_occurrence_id
FROM
(
    SELECT
        v.person_id,
        v.observation_period_start_date,
        SUM(CASE WHEN v.visit_concept_id IN {visit_concept_ids} THEN 1 ELSE 0 END) AS num_of_hospitalizations
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
        WHERE op.observation_period_start_date >= '{date_lower_bound}' 
            AND op.observation_period_start_date <= '{date_upper_bound}'
            AND DATE_ADD(op.observation_period_start_date, {total_window}) <= op.observation_period_end_date
    ) v
    GROUP BY v.observation_period_start_date, v.person_id
) v
WHERE v.num_of_hospitalizations = 0
"""

VISIT_CONCEPT_IDS = (9201, 262)

DOMAIN_TABLE_LIST = ['condition_occurrence', 'drug_exposure', 'procedure_occurrence']

PERSON = 'person'
VISIT_OCCURRENCE = 'visit_occurrence'
OBSERVATION_PERIOD = 'observation_period'
DEPENDENCY_LIST = [PERSON, VISIT_OCCURRENCE, OBSERVATION_PERIOD]

HOSPITALIZATION_TARGET = 'hospitalization_target'
HOSPITALIZATION_OUTCOME = 'hospitalization_outcome'


class HospitalizationTargetCohortBuilder(BaseCohortBuilder):

    def preprocess_dependencies(self):
        pass

    def create_cohort(self):
        # The qualifying patients can't have any hospitalization record before observation_window
        # plus hold_off_window
        first_qualified_visit_query = TARGET_COHORT_QUERY_TEMPLATE.format(
            visit_concept_ids=VISIT_CONCEPT_IDS,
            total_window=self._post_observation_period,
            date_lower_bound=self._date_lower_bound,
            date_upper_bound=self._date_upper_bound
        )

        cohort = self.spark.sql(first_qualified_visit_query)

        return cohort


class HospitalizationOutcomeCohortBuilder(HospitalizationTargetCohortBuilder):

    def create_cohort(self):
        cohort = self.spark.sql(
            OUTCOME_COHORT_QUERY_TEMPLATE.format(visit_concept_ids=VISIT_CONCEPT_IDS))
        return cohort


def main(cohort_name, input_folder, output_folder, date_lower_bound, date_upper_bound,
         age_lower_bound, age_upper_bound, observation_window, prediction_window, hold_off_window,
         index_date_match_window, include_visit_type, is_feature_concept_frequency,
         is_roll_up_concept):
    # Generate the target cohort
    target_cohort = HospitalizationTargetCohortBuilder(
        HOSPITALIZATION_TARGET,
        input_folder,
        output_folder,
        date_lower_bound,
        date_upper_bound,
        age_lower_bound,
        age_upper_bound,
        prior_observation_period=0,
        post_observation_period=observation_window + hold_off_window,
        dependency_list=DEPENDENCY_LIST).build().load_cohort()

    # Generate the outcome cohort
    outcome_cohort = HospitalizationOutcomeCohortBuilder(
        HOSPITALIZATION_OUTCOME,
        input_folder,
        output_folder,
        date_lower_bound,
        date_upper_bound,
        age_lower_bound,
        age_upper_bound,
        prior_observation_period=0,
        post_observation_period=0,
        dependency_list=DEPENDENCY_LIST).build().load_cohort()

    NestedCohortBuilder(cohort_name=cohort_name,
                        input_folder=input_folder,
                        output_folder=output_folder,
                        target_cohort=target_cohort,
                        outcome_cohort=outcome_cohort,
                        ehr_table_list=DOMAIN_TABLE_LIST,
                        observation_window=observation_window,
                        hold_off_window=hold_off_window,
                        prediction_start_days=0,
                        prediction_window=prediction_window,
                        is_window_post_index=True,
                        include_ehr_records=True,
                        include_visit_type=include_visit_type,
                        is_feature_concept_frequency=is_feature_concept_frequency,
                        is_roll_up_concept=is_roll_up_concept).build()


if __name__ == '__main__':
    spark_args = create_spark_args()

    main(spark_args.cohort_name,
         spark_args.input_folder,
         spark_args.output_folder,
         spark_args.date_lower_bound,
         spark_args.date_upper_bound,
         spark_args.lower_bound,
         spark_args.upper_bound,
         spark_args.observation_window,
         spark_args.prediction_window,
         spark_args.hold_off_window,
         spark_args.index_date_match_window,
         spark_args.include_visit_type,
         spark_args.is_feature_concept_frequency,
         spark_args.is_roll_up_concept)
