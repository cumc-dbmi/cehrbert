from spark_apps.cohorts.spark_app_base import create_prediction_cohort
from spark_apps.cohorts import death
from spark_apps.cohorts import last_visit_discharged_home as last

from spark_apps.parameters import create_spark_args

DOMAIN_TABLE_LIST = ['condition_occurrence', 'drug_exposure', 'procedure_occurrence']

if __name__ == '__main__':
    create_prediction_cohort(create_spark_args(),
                             last.query_builder(),
                             death.query_builder(),
                             DOMAIN_TABLE_LIST)
