from abc import ABC, abstractmethod
import copy
import os
import logging
from typing import Union, List, Dict
from tqdm import tqdm
from docarray import BaseDoc, DocList
from docarray.index.abstract import BaseDocIndex
import numpy as np
from dask.dataframe import DataFrame as dd_dataframe
from pandas import DataFrame as pd_dataframe
from data_generators.tokenizer import ConceptTokenizer

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("PatientDataIndex")


class PatientDataIndex(ABC):
    def __init__(
            self,
            concept_tokenizer: ConceptTokenizer,
            index_folder: str = None,
            rebuilt: bool = False,
            incremental_built: bool = False,
            set_unique_concepts: bool = False,
            common_attributes: List[str] = None,
            sensitive_attributes: List[str] = None,
            batch_size: int = 1024
    ):
        self.concept_tokenizer = concept_tokenizer
        self.index_folder = index_folder
        self.rebuilt = rebuilt
        self.incremental_built = incremental_built
        self.set_unique_concepts = set_unique_concepts
        self.common_attributes = common_attributes
        self.sensitive_attributes = sensitive_attributes
        self.batch_size = batch_size

        LOG.info(
            f'The {self.__class__.__name__} parameters\n'
            f'\tconcept_tokenizer: {concept_tokenizer}\n'
            f'\tindex_folder: {index_folder}\n'
            f'\trebuilt: {rebuilt}\n'
            f'\tincremental_built: {incremental_built}\n'
            f'\tset_unique_concepts: {set_unique_concepts}\n'
            f'\tcommon_attributes: {common_attributes}\n'
            f'\tsensitive_attributes: {sensitive_attributes}\n'
            f'\tbatch_size: {batch_size}\n'
        )

        # Get the index for search
        self.doc_class = self.create_doc_class()
        self.doc_index = self.create_index()
        self.is_sensitive_attributes_str = (
                self.doc_class.model_json_schema()['properties']['sensitive_attributes']['type'] == 'string'
        )
        self.is_concepts_str = (
                self.doc_class.model_json_schema()['properties']['concept_ids']['type'] == 'string'
        )

    @abstractmethod
    def delete_index(self):
        pass

    @abstractmethod
    def get_all_person_ids(self) -> List[int]:
        pass

    @abstractmethod
    def create_doc_class(self) -> BaseDoc:
        pass

    @abstractmethod
    def create_index(self) -> BaseDocIndex[BaseDoc]:
        pass

    @abstractmethod
    def find_patients(
            self,
            year: int,
            age: int,
            gender: str,
            race: str,
            concept_ids: List[str],
            year_std: int,
            age_std: int,
            limit: int
    ) -> List[Dict]:
        pass

    def is_index_empty(self):
        return self.doc_index.num_docs() == 0

    def build_index(
            self,
            dataset: Union[pd_dataframe, dd_dataframe]
    ):
        LOG.info('Started adding documents to the index.')

        if not self.is_index_empty():
            if self.rebuilt:
                LOG.info('Rebuilt is specified. Started deleting documents from the index.')
                self.delete_index()
                LOG.info('Done deleting documents from the index.')
            elif not self.incremental_built:
                raise RuntimeError(
                    f'The index already exists in {self.index_folder}. '
                    f'If you want to recreate the index, set --rebuilt'
                )

        # incremental built and skip the person_ids that exist in the index already
        if self.incremental_built:
            existing_person_ids = self.get_all_person_ids()
            dataset = dataset[~dataset.person_id.isin(existing_person_ids)]

        batch_of_docs = []
        for t in tqdm(dataset.itertuples(), total=len(dataset)):

            if self.validate_demographics(t.concept_ids):
                person_id = str(t.person_id)
                year, age, gender, race = self.get_demographics(t.concept_ids)
                medical_concepts = self.extract_medical_concepts(t.concept_ids)
                common_concepts = copy.deepcopy(medical_concepts)
                sensitive_concepts = []

                if self.common_attributes and self.sensitive_attributes:
                    common_concepts = [_ for _ in medical_concepts if _ in self.common_attributes]
                    sensitive_concepts = [_ for _ in medical_concepts if _ in self.sensitive_attributes]

                # If the common concepts is empty, we will skip this example
                if len(common_concepts) == 0:
                    continue

                embeddings = self.create_binary_format(common_concepts)

                if self.is_sensitive_attributes_str:
                    sensitive_concepts = ','.join(sensitive_concepts)

                if self.is_concepts_str:
                    common_concepts = ','.join(common_concepts)

                document = self.doc_class(
                    id=person_id,
                    person_id=person_id,
                    year=year,
                    age=age,
                    gender=gender,
                    race=race,
                    concept_embeddings=embeddings,
                    sensitive_attributes=sensitive_concepts,
                    concept_ids=common_concepts,
                    num_of_visits=t.num_of_visits,
                    num_of_concepts=t.num_of_concepts

                )
                batch_of_docs.append(document)

                if batch_of_docs and len(batch_of_docs) % self.batch_size == 0:
                    docs = DocList[self.doc_class](batch_of_docs)
                    self.doc_index.index(docs)
                    LOG.info('Done adding documents.')
                    batch_of_docs.clear()

        if batch_of_docs:
            docs = DocList[self.doc_class](batch_of_docs)
            self.doc_index.index(docs)
            LOG.info('Done adding the final batch documents.')

    def create_binary_format(self, concept_ids):
        indices = np.array(self.concept_tokenizer.encode(concept_ids)).flatten()
        embeddings = np.zeros(self.concept_tokenizer.get_vocab_size())
        embeddings.put(indices, 1)
        return embeddings

    def search(
            self,
            patient_seq: List[str],
            year_std: int = 1,
            age_std: int = 1,
            limit: int = 1
    ) -> List[Dict]:

        if self.is_index_empty():
            LOG.warning(f'The index is empty, please build the index first!')
            return None

        if not self.validate_demographics(patient_seq):
            LOG.warning(f'The first four tokens {patient_seq[0:4]} do not contain valid demographic information!')
            return None

        year, age, gender, race = self.get_demographics(patient_seq)
        concept_ids = self.extract_medical_concepts(patient_seq)

        if self.common_attributes:
            concept_ids = [_ for _ in concept_ids if _ in self.common_attributes]

        return self.find_patients(
            year, age, gender, race, concept_ids, year_std, age_std, limit
        )

    def extract_medical_concepts(self, patient_seq):
        concept_ids = [_ for _ in patient_seq[4:] if str.isnumeric(_)]
        if self.set_unique_concepts:
            concept_ids = list(set(concept_ids))
        return concept_ids

    @staticmethod
    def validate_demographics(
            concept_ids
    ):
        year_token, age_token, gender, race = concept_ids[0:4]
        if year_token[:5] != 'year:':
            return False
        if age_token[:4] != 'age:':
            return False
        return True

    @staticmethod
    def get_demographics(
            concept_ids
    ):
        year_token, age_token, gender, race = concept_ids[0:4]
        try:
            year = int(year_token[5:])
        except ValueError as e:
            LOG.error(f'{year_token[5:]} cannot be converted to an integer, use the default value 1900')
            year = 1900

        try:
            age = int(age_token[4:])
        except ValueError as e:
            LOG.error(f'{age_token[4:]} cannot be converted to an integer, use the default value 1900')
            age = -1

        return year, age, gender, race
