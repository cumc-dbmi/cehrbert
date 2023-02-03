# Hierarchical CEHR-BERT
This project is the continuation of the CEHR-BERT work, which has been published at https://proceedings.mlr.press/v158/pang21a.html. 

## CEHR-BERT Architecture
Hierarchical CEHR-BERT currently only supports the structured EHR data in the OMOP format, which is a common data model used to support observational studies and managed  by the Observational Health Data Science and Informatics (OHDSI) open-science community. 
There are three major components in CEHR-BERT, data generation, model pre-training, and model evaluation with fine-tuning, those components work in conjunction to provide an end-to-end model evaluation framework.

## Getting Started


### Pre-requisite
The project is built in python 3.7, and project dependency needs to be installed 

`pip3 install -r requirements.txt`

Create the following folders for the tutorial below
```console
mkdir -p ~/Documents/omop_test/hierarchical_bert;
mkdir -p ~/Documents/omop_test/cehr-bert;
```

### 1. Download OMOP tables as parquet files
We have created a spark app to download OMOP tables from Sql Server as parquet files. You need adjust the properties in `db_properties.ini` to match with your database setup.
```console
spark-submit tools/download_omop_tables.py -c db_properties.ini -tc person visit_occurrence condition_occurrence procedure_occurrence drug_exposure measurement observation_period concept concept_relationship concept_ancestor -o ~/Documents/omop_test/
```
### 2. Generate training data for Hierarchical CEHR-BERT
#### Hierarchical training data
This approach views the patient history as a list of visits, where each visit is a group of medical events. We order the patient events in chronological order and construct a list of lists, where the sublist contains all medical events associated with the same visit
```console
spark-submit spark_apps/generate_hierarchical_bert_training_data.py  -i ~/Documents/omop_test/  -o ~/Documents/omop_test/hierarchical_bert -tc condition_occurrence procedure_occurrence drug_exposure -d 1985-01-01 
```
#### CEHR-BERT training data
We order the patient events in chronological order and put all data points in a sequence. This approach allows us to apply BERT to structured EHR as-is. 
```console
spark-submit spark_apps/generate_training_data.py -i ~/Documents/omop_test/ -o ~/Documents/omop_test/cehr-bert -tc condition_occurrence procedure_occurrence drug_exposure -d 1985-01-01 --is_new_patient_representation -iv 
```

### 3. Pre-train Hierarchical CEHR-BERT
#### Train Hierarchical CEHR-BERT
```console
PYTHONPATH=./: python3 trainers/train_probabilistic_phenotype.py -i ~/Documents/omop_test/hierarchical_bert -o ~/Documents/omop_test/hierarchical_bert -b 32 --max_num_visits 20 --max_num_concepts 50 -e 1 -d 2 -iv --include_att_prediction --include_readmission --num_of_phenotypes 100 --num_of_concept_neighbors 160
```

#### Train CEHR-BERT
```console
PYTHONPATH=./: python3 trainers/train_bert_only.py -i ~/Documents/omop_test/cehr-bert -o ~/Documents/omop_test/cehr-bert -iv -m 512 -e 1 -b 32 -d 5 
```
### 4. Generate hf readmission prediction task
#### Generate hf_readmission prediction data for Hierarchical CEHR-BERT
```console
PYTHONPATH=./:$PYTHONPATH spark-submit spark_apps/prediction_cohorts/hf_readmission.py -c hf_readmission -i ~/Documents/omop_test/ -o ~/Documents/omop_test/hierarchical_bert -dl 1985-01-01 -du 2020-12-31 -l 18 -u 100 -ow 360 -ps 0 -pw 30 --is_hierarchical_bert
```

#### Generate hf_readmission prediction data for CEHR-BERT
```console
PYTHONPATH=./:$PYTHONPATH spark-submit spark_apps/prediction_cohorts/hf_readmission.py -c hf_readmission -i ~/Documents/omop_test/ -o ~/Documents/omop_test/cehr-bert -dl 1985-01-01 -du 2020-12-31 -l 18 -u 100 -ow 360 -ps 0 -pw 30 --is_new_patient_representation
```

### 5. Fine-tune Hierarchical CEHR-BERT for hf readmission
#### Fine-tune Hierarchical CEHR-BERT for the hf readmission prediction
```console
# Copy the pretrained bert model
cp ~/Documents/omop_test/hierarchical_bert/bert_model_01_* ~/Documents/omop_test/hierarchical_bert/bert_model.h5
PYTHONPATH=./: python3 evaluations/evaluation.py -a sequence_model -sd ~/Documents/omop_test/hierarchical_bert/hf_readmission -ef ~/Documents/omop_test/evaluation_train_val_split/hf_readmission/ -m 1 -b 32 -p 10 -vb ~/Documents/omop_test/hierarchical_bert -me hierarchical_bert_lstm --sequence_model_name hierarchical_bert_with_phenotype_cross_validation_test --max_num_of_visits 20 --max_num_of_concepts 50 --num_of_folds 3 --cross_validation_test --grid_search_config full_grid_search_config.ini
```

#### Fine-tune CEHR-BERT for the hf readmission prediction
```console
# Copy the pretrained bert model
cp ~/Documents/omop_test/cehr-bert/bert_model_01_* ~/Documents/omop_test/cehr-bert/bert_model.h5;
PYTHONPATH=./: python3 evaluations/evaluation.py -a sequence_model -sd ~/Documents/omop_test/cehr-bert/hf_readmission -ef ~/Documents/omop_test/evaluation_train_val_split/hf_readmission/ -m 512 -b 32 -p 10 -vb ~/Documents/omop_test/cehr-bert -me vanilla_bert_lstm --sequence_model_name CEHR_BERT_512_cross_validation_test --num_of_folds 3 --learning_rate 1e-4 --cross_validation_test --grid_search_config full_grid_search_config.ini;
```