from spark_apps.spark_parse_args import create_spark_args
from spark_apps.cohorts import type_two_diabietes as t2dm
from spark_apps.cohorts import heart_failure as hf
from spark_apps.cohorts.spark_app_base import create_prediction_cohort

DOMAIN_TABLE_LIST = ['condition_occurrence', 'drug_exposure',
                     'procedure_occurrence', 'measurement']

if __name__ == '__main__':
    spark_args = create_spark_args()

    ehr_table_list = spark_args.ehr_table_list if spark_args.ehr_table_list else DOMAIN_TABLE_LIST

    create_prediction_cohort(
        spark_args,
        t2dm.query_builder(),
        hf.query_builder(),
        ehr_table_list
    )
