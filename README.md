## CEHR-BERT Architecture
CEHR-BERT is a large language model developed for the structured EHR data, the work has been published at https://proceedings.mlr.press/v158/pang21a.html. CEHR-BERT currently only supports the structured EHR data in the OMOP format, which is a common data model used to support observational studies and managed  by the Observational Health Data Science and Informatics (OHDSI) open-science community. 
There are three major components in CEHR-BERT, data generation, model pre-training, and model evaluation with fine-tuning, those components work in conjunction to provide an end-to-end model evaluation framework.

## Getting Started


### Pre-requisite
The project is built in python 3.7, and project dependency needs to be installed 

`pip3 install -r requirements.txt`

Create the following folders for the tutorial below
```console
mkdir -p ~/Documents/omop_test/cehr-bert;
```

### 1. Download OMOP tables as parquet files
We have created a spark app to download OMOP tables from Sql Server as parquet files. You need adjust the properties in `db_properties.ini` to match with your database setup.
```console
spark-submit tools/download_omop_tables.py -c db_properties.ini -tc person visit_occurrence condition_occurrence procedure_occurrence drug_exposure measurement observation_period concept concept_relationship concept_ancestor -o ~/Documents/omop_test/
```
### 2. Generate training data for CEHR-BERT
We order the patient events in chronological order and put all data points in a sequence. This approach allows us to apply BERT to structured EHR as-is. 
```console
spark-submit spark_apps/generate_training_data.py -i ~/Documents/omop_test/ -o ~/Documents/omop_test/cehr-bert -tc condition_occurrence procedure_occurrence drug_exposure -d 1985-01-01 --is_new_patient_representation -iv 
```

### 3. Pre-train CEHR-BERT
If you don't have your own OMOP instance, we have provided a sample of patient sequence data generated using Synthea at `sample/patient_sequence` in the repo. CEHR-BERT expects the data folder to be named as `patient_sequence` and placed at `~/Documents/omop_test/patient_sequence` in this example
```console
PYTHONPATH=./: python3 trainers/train_bert_only.py -i ~/Documents/omop_test/ -o ~/Documents/omop_test/cehr-bert -iv -m 512 -e 1 -b 32 -d 5 
```
### 4. Generate hf readmission prediction task
```console
PYTHONPATH=./:$PYTHONPATH spark-submit spark_apps/prediction_cohorts/hf_readmission.py -c hf_readmission -i ~/Documents/omop_test/ -o ~/Documents/omop_test/cehr-bert -dl 1985-01-01 -du 2020-12-31 -l 18 -u 100 -ow 360 -ps 0 -pw 30 --is_new_patient_representation
```

### 5. Fine-tune CEHR-BERT for hf readmission
```console
# Copy the pretrained bert model
cp ~/Documents/omop_test/cehr-bert/bert_model_01_* ~/Documents/omop_test/cehr-bert/bert_model.h5;
PYTHONPATH=./: python3 evaluations/evaluation.py -a sequence_model -sd ~/Documents/omop_test/cehr-bert/hf_readmission -ef ~/Documents/omop_test/evaluation_train_val_split/hf_readmission/ -m 512 -b 32 -p 10 -vb ~/Documents/omop_test/cehr-bert -me vanilla_bert_lstm --sequence_model_name CEHR_BERT_512_cross_validation_test --num_of_folds 3 --learning_rate 1e-4 --cross_validation_test --grid_search_config full_grid_search_config.ini;
```
