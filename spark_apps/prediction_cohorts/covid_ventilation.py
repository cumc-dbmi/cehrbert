from spark_apps.cohorts import covid, ventilation
from spark_apps.cohorts.spark_app_base import create_prediction_cohort

from spark_apps.spark_parse_args import create_spark_args

DOMAIN_TABLE_LIST = ['condition_occurrence', 'drug_exposure', 'procedure_occurrence']

if __name__ == '__main__':
    create_prediction_cohort(create_spark_args(),
                             covid.query_builder(),
                             ventilation.query_builder(),
                             DOMAIN_TABLE_LIST)
