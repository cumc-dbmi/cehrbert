from spark_apps.cohorts.spark_app_base import create_prediction_cohort
from spark_apps.cohorts import death
from spark_apps.cohorts import last_visit_discharged_home as last

from spark_apps.spark_parse_args import create_spark_args

DOMAIN_TABLE_LIST = ['condition_occurrence', 'drug_exposure', 'procedure_occurrence', 'measurement']

if __name__ == '__main__':
    spark_args = create_spark_args()
    ehr_table_list = spark_args.ehr_table_list if spark_args.ehr_table_list else DOMAIN_TABLE_LIST

    create_prediction_cohort(
        spark_args,
        last.query_builder(spark_args),
        death.query_builder(),
        ehr_table_list
    )
