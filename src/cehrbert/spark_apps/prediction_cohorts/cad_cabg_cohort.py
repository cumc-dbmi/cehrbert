from ..spark_parse_args import create_spark_args
from ..cohorts import coronary_artery_disease as cad
from ..cohorts import cabg
from ..cohorts.spark_app_base import create_prediction_cohort

DOMAIN_TABLE_LIST = ["condition_occurrence", "drug_exposure", "procedure_occurrence"]

if __name__ == "__main__":
    spark_args = create_spark_args()

    ehr_table_list = (
        spark_args.ehr_table_list if spark_args.ehr_table_list else DOMAIN_TABLE_LIST
    )

    create_prediction_cohort(
        spark_args,
        cad.query_builder(spark_args),
        cabg.query_builder(spark_args),
        ehr_table_list,
    )
