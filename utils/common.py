from os import path

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import Window

SUB_WINDOW_SIZE = 30

NUM_OF_PARTITIONS = 600

DOMAIN_KEY_FIELDS = {
    'condition_occurrence_id': ('condition_concept_id', 'condition_start_date', 'condition'),
    'procedure_occurrence_id': ('procedure_concept_id', 'procedure_date', 'procedure'),
    'drug_exposure_id': ('drug_concept_id', 'drug_exposure_start_date', 'drug'),
    'measurement_id': ('measurement_concept_id', 'measurement_date', 'measurement')
}


# +
def get_key_fields(domain_table):
    field_names = domain_table.schema.fieldNames()
    for k, v in DOMAIN_KEY_FIELDS.items():
        if k in field_names:
            return v
    return (get_concept_id_field(domain_table), get_domain_date_field(domain_table), get_domain_field(domain_table))


def get_domain_date_field(domain_table):
    # extract the domain start_date column
    return [f for f in domain_table.schema.fieldNames() if 'date' in f][0]


def get_concept_id_field(domain_table):
    return [f for f in domain_table.schema.fieldNames() if 'concept_id' in f][0]


def get_domain_field(domain_table):
    return get_concept_id_field(domain_table).replace('_concept_id', '')


# +
def create_file_path(input_folder, table_name):
    if input_folder[-1] == '/':
        file_path = input_folder + table_name
    else:
        file_path = input_folder + '/' + table_name

    return file_path


def get_patient_event_folder(output_folder):
    return create_file_path(output_folder, 'patient_event')


def get_patient_sequence_folder(output_folder):
    return create_file_path(output_folder, 'patient_sequence')


def get_patient_sequence_csv_folder(output_folder):
    return create_file_path(output_folder, 'patient_sequence_csv')


def get_pairwise_euclidean_distance_output(output_folder):
    return create_file_path(output_folder, 'pairwise_euclidean_distance.pickle')


def get_pairwise_cosine_similarity_output(output_folder):
    return create_file_path(output_folder, 'pairwise_cosine_similarity.pickle')


def write_sequences_to_csv(spark, patient_sequence_path, patient_sequence_csv_path):
    spark.read.parquet(patient_sequence_path).select('concept_list').repartition(1) \
        .write.mode('overwrite').option('header', 'false').csv(patient_sequence_csv_path)


# -
def join_domain_time_span(domain_tables, span=0):
    """Standardize the format of OMOP domain tables using a time frame
    
    Keyword arguments:
    domain_tables -- the array containing the OMOOP domain tabls except visit_occurrence
    span -- the span of the time window
    
    The the output columns of the domain table is converted to the same standard format as the following 
    (person_id, standard_concept_id, date, lower_bound, upper_bound, domain). 
    In this case, co-occurrence is defined as those concept ids that have co-occurred 
    within the same time window of a patient.
    
    """
    patient_event = None

    for domain_table in domain_tables:
        # extract the domain concept_id from the table fields. E.g. condition_concept_id from condition_occurrence
        # extract the domain start_date column
        # extract the name of the table
        concept_id_field, date_field, table_domain_field = get_key_fields(domain_table)

        domain_table = domain_table.withColumn("date", F.to_date(F.col(date_field))) \
            .withColumn("lower_bound", F.date_add(F.col(date_field), -span)) \
            .withColumn("upper_bound", F.date_add(F.col(date_field), span))

        # standardize the output columns
        domain_table = domain_table.where(F.col(concept_id_field).cast('string') != '0') \
            .select(domain_table["person_id"],
                    domain_table[concept_id_field].alias("standard_concept_id"),
                    domain_table["date"],
                    domain_table["lower_bound"],
                    domain_table["upper_bound"],
                    domain_table['visit_occurrence_id'],
                    F.lit(table_domain_field).alias("domain")) \
            .distinct()

        if patient_event == None:
            patient_event = domain_table
        else:
            patient_event = patient_event.union(domain_table)

    return patient_event


def join_domain_tables(domain_tables):
    """Standardize the format of OMOP domain tables using a time frame
    
    Keyword arguments:
    domain_tables -- the array containing the OMOOP domain tabls except visit_occurrence
    
    The the output columns of the domain table is converted to the same standard format as the following 
    (person_id, standard_concept_id, date, lower_bound, upper_bound, domain). 
    In this case, co-occurrence is defined as those concept ids that have co-occurred 
    within the same time window of a patient.
    
    """
    patient_event = None

    for domain_table in domain_tables:
        # extract the domain concept_id from the table fields. E.g. condition_concept_id from condition_occurrence
        # extract the domain start_date column
        # extract the name of the table
        concept_id_field, date_field, table_domain_field = get_key_fields(domain_table)
        # standardize the output columns
        domain_table = domain_table.where(F.col(concept_id_field).cast('string') != '0') \
            .withColumn('date', F.to_date(F.col(date_field)))

        domain_table = domain_table.select(domain_table['person_id'],
                                           domain_table[concept_id_field].alias('standard_concept_id'),
                                           domain_table['date'],
                                           domain_table['visit_occurrence_id'],
                                           F.lit(table_domain_field).alias('domain')) \
            .distinct()

        if patient_event == None:
            patient_event = domain_table
        else:
            patient_event = patient_event.union(domain_table)

    return patient_event


def preprocess_domain_table(spark, input_folder, domain_table_name):
    domain_table = spark.read.parquet(create_file_path(input_folder, domain_table_name))
    # lowercase the schema fields
    domain_table = domain_table.select([F.col(f_n).alias(f_n.lower()) for f_n in domain_table.schema.fieldNames()])
    _, _, domain_field = get_key_fields(domain_table)

    if domain_field == 'drug' \
            and path.exists(create_file_path(input_folder, 'concept')) \
            and path.exists(create_file_path(input_folder, 'concept_ancestor')):
        concept = spark.read.parquet(create_file_path(input_folder, 'concept'))
        concept_ancestor = spark.read.parquet(create_file_path(input_folder, 'concept_ancestor'))
        domain_table = roll_up_to_drug_ingredients(domain_table, concept, concept_ancestor)

    return domain_table


def roll_up_to_drug_ingredients(drug_exposure, concept, concept_ancestor):
    # lowercase the schema fields
    drug_exposure = drug_exposure.select([F.col(f_n).alias(f_n.lower()) for f_n in drug_exposure.schema.fieldNames()])

    drug_ingredient = drug_exposure.select('drug_concept_id').distinct() \
        .join(concept_ancestor, F.col('drug_concept_id') == F.col('descendant_concept_id')) \
        .join(concept, F.col('ancestor_concept_id') == F.col('concept_id')) \
        .where(concept['concept_class_id'] == 'Ingredient') \
        .select(F.col('drug_concept_id'), F.col('concept_id').alias('ingredient_concept_id'))

    drug_ingredient_fields = [
        F.coalesce(F.col('ingredient_concept_id'), F.col('drug_concept_id')).alias('drug_concept_id')]
    drug_ingredient_fields.extend(
        [F.col(field_name) for field_name in drug_exposure.schema.fieldNames() if field_name != 'drug_concept_id'])

    drug_exposure = drug_exposure.join(drug_ingredient, 'drug_concept_id', 'left_outer') \
        .select(drug_ingredient_fields)

    return drug_exposure
