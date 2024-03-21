from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
import os


def creat_global_temp(folder, spark, name_prefix):
    for df_name in ['person', 'condition_occurrence', 'concept', 'concept_ancestor']:
        df = spark.read.parquet(os.path.join(folder, df_name))
        df.createOrReplaceGlobalTempView(name_prefix + '_' + df_name)
    return f"global temp views for {folder} have been created"


def get_common_concepts(spark: SparkSession, common_concept_percent):
    target_event = spark.sql("""select condition_concept_id,
        ancestor_concept_id,
        count(distinct person_id) as concept_pt_cnt
        from
        (
        select distinct co.condition_concept_id, min(ca.ancestor_concept_id) as ancestor_concept_id,
        co.person_id
        from global_temp.omop_condition_occurrence co
        join global_temp.omop_concept c
        on co.condition_concept_id = c.concept_id
        join global_temp.omop_concept_ancestor ca
        on c.concept_id = ca.descendant_concept_id
        where ca.max_levels_of_separation = 3
        group by co.condition_concept_id, co.person_id
        ) a
        group by a.condition_concept_id, a.ancestor_concept_id""")
    total_pt_cnt = spark.sql("""
        select count(distinct person_id) as cnt
        from global_temp.omop_condition_occurrence""")

    target_event_filter = target_event.where(F.col('concept_pt_cnt') >= common_concept_percent * total_pt_cnt.collect()[0][0])
    target_common_concept = target_event_filter.select(F.col('condition_concept_id').alias('common_condition_concept_id'))
    return target_common_concept



