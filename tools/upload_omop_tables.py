import configparser
import argparse
from pathlib import Path
from pyspark.context import SparkContext
import pyspark.sql.functions as f

sc = SparkContext()
# Get current sparkconf which is set by glue
conf = sc.getConf()
# add additional spark configurations
conf.set("spark.sql.legacy.parquet.int96RebaseModeInRead", "CORRECTED")
conf.set("spark.sql.legacy.parquet.int96RebaseModeInWrite", "CORRECTED")
conf.set("spark.sql.legacy.parquet.datetimeRebaseModeInRead", "CORRECTED")
conf.set("spark.sql.legacy.parquet.datetimeRebaseModeInWrite", "CORRECTED")
# Restart spark context
sc.stop()


omop_table_dict = {'person': 'person_id',
                   'condition_occurrence': 'condition_occurrence_id',
                   'measurement': 'measurement_id',
                   'drug_exposure': 'drug_exposure_id',
                   'procedure_occurrence': 'procedure_occurrence_id',
                   'observation': 'observation_id',
                   'visit_occurrence': 'visit_occurrence_id'}
omop_timestamp_dict = {'person': 'birth_datetime',
                       'condition_occurrence': 'condition_start_date',
                       'measurement': 'measurement_date',
                       'drug_exposure': 'drug_exposure_start_date',
                       'procedure_occurrence': 'procedure_date',
                       'observation': 'observation_date',
                       'visit_occurrence': 'visit_start_date'}

sc = SparkContext.getOrCreate(conf=conf)
from pyspark.sql import SQLContext

sqlContext = SQLContext(sc)


def upload_omop_tables(
        domain_table_folder,
        db_properties
):
    df = sqlContext.read.format('parquet').load(str(domain_table_folder)+'/')
    df = df.filter(f.col(omop_timestamp_dict[domain_table_folder.name]) > f.unix_timestamp(f.lit('1900-01-01 00:00:00')).cast('timestamp'))
    df = df.filter(f.col(omop_timestamp_dict[domain_table_folder.name]) < f.unix_timestamp(f.lit('9999-01-01 00:00:00')).cast('timestamp'))
    
    # cast the concept id columns to integer type
    for column in df.columns:
        if 'concept_id' in column:
            df = df.withColumn(column, f.col(column).cast('integer'))
        if 'date' in column:
            df = df.withColumn(column, f.col(column).cast('date'))

    #if domain_table_folder.name == 'person':
    #    df = df.withColumn("birth_datetime", df["birth_datetime"].cast("string"))
    df.repartition(10).write.format('jdbc').options(
      url=db_properties['base_url'],
      dbtable=f"{domain_table_folder.name}",
      user=db_properties['user'],
      password=db_properties['password'],
      batchsize=200000,
      queryTimeout=500
      ).mode("overwrite").save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for uploading OMOP tables')

    parser.add_argument('-c',
                        '--credential_path',
                        dest='credential_path',
                        action='store',
                        help='The path for your database credentials',
                        required=True)

    parser.add_argument('-i',
                        '--input_folder',
                        dest='input_folder',
                        action='store',
                        help=('Full path to the input folder that contains '
                              'the domain tables as separate folders with parquet files'),
                        required=True)

    ARGS = parser.parse_args()
    credential_path = ARGS.credential_path
    input_folder = Path(ARGS.input_folder)
    config = configparser.ConfigParser()
    config.read(credential_path)
    properties = config.defaults()
    uploaded_tables = []

    for folder in input_folder.glob('*'):
        try:
            if folder.name in omop_table_dict:
                upload_omop_tables(folder, properties)
                uploaded_tables.append(folder.name)
                print(f'Table: {str(folder.name)} is uploaded')
        except Exception as e:
            print(str(e))
            print(f'The table {str(folder.name)} was not uploaded')
    print(f'The following tables were uploaded:{str(uploaded_tables)}')
