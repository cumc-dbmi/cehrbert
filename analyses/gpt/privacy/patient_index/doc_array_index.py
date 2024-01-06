import copy
import os
import shutil
import logging
from typing import Union, List, Generator, Dict, Type
from tqdm import tqdm
from data_generators.tokenizer import ConceptTokenizer
from docarray import BaseDoc, DocList
from docarray.index import HnswDocumentIndex
from docarray.typing import NdArray
import numpy as np
from dask.dataframe import DataFrame as dd_dataframe
from pandas import DataFrame as pd_dataframe

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("PatientDataIndex")


# Factory function to create a class with a dynamic embedding size
def create_doc_class(vocab_size: int) -> Type[BaseDoc]:
    class PatientDocument(BaseDoc):
        id: str
        person_id: str
        year: int
        age: int
        race: str
        gender: str
        concept_embeddings: NdArray[vocab_size]
        sensitive_attributes: List[str]
        num_of_visits: int
        num_of_concepts: int

    return PatientDocument


class PatientDataIndex:
    def __init__(
            self,
            index_folder: str,
            concept_tokenizer: ConceptTokenizer,
            rebuilt: bool = False,
            set_unique_concepts: bool = False,
            common_attributes: List[str] = None,
            sensitive_attributes: List[str] = None
    ):
        self.index_folder = index_folder
        self.concept_tokenizer = concept_tokenizer
        self.rebuilt = rebuilt
        self.set_unique_concepts = set_unique_concepts
        self.common_attributes = common_attributes
        self.sensitive_attributes = sensitive_attributes

        LOG.info(
            f'The PatientDataIndex parameters\n'
            f'\tindex_folder: {index_folder}\n'
            f'\trebuilt: {rebuilt}\n'
            f'\tconcept_tokenizer: {concept_tokenizer}\n'
            f'\tset_unique_concepts: {set_unique_concepts}\n'
            f'\tcommon_attributes: {common_attributes}\n'
            f'\tsensitive_attributes: {sensitive_attributes}\n'
        )

        # Get the index for search
        self.doc_class = create_doc_class(self.concept_tokenizer.get_vocab_size())
        self.doc_index = HnswDocumentIndex[self.doc_class](work_dir=self.index_folder)

    def is_index_empty(self):
        return self.doc_index.num_docs() == 0

    def build_index(
            self,
            dataset: Union[pd_dataframe, dd_dataframe]
    ):
        LOG.info('Started adding documents to the index.')

        # Create a directory to store the index
        if not os.path.exists(self.index_folder):
            LOG.info(f'The index folder {self.index_folder} does not exist, creating it')
            os.mkdir(self.index_folder)

        if not self.is_index_empty():
            if self.rebuilt:
                try:
                    shutil.rmtree(self.index_folder)
                    os.mkdir(self.index_folder)
                    LOG.info(f"The directory {self.index_folder} and all its contents have been removed")
                    self.doc_index = HnswDocumentIndex[self.doc_class](work_dir=self.index_folder)
                except OSError as error:
                    LOG.error(f"Error: {error}")
                    LOG.error(f"Could not remove the directory {self.index_folder}")
                    raise error
            else:
                raise RuntimeError(
                    f'The index already exists in {self.index_folder}. If you want to recreate the index, set --rebuilt'
                )

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

                embeddings = self.create_binary_format(common_concepts)

                document = self.doc_class(
                    id=person_id,
                    person_id=person_id,
                    year=year,
                    age=age,
                    gender=gender,
                    race=race,
                    concept_embeddings=embeddings,
                    sensitive_attributes=sensitive_concepts,
                    num_of_visits=t.num_of_visits,
                    num_of_concepts=t.num_of_concepts

                )
                batch_of_docs.append(document)

                if batch_of_docs and len(batch_of_docs) % 10240 == 0:
                    docs = DocList[self.doc_class](batch_of_docs)
                    self.doc_index.index(docs)
                    LOG.info('Done adding documents.')
                    batch_of_docs.clear()

        if batch_of_docs:
            docs = DocList[self.doc_class](batch_of_docs)
            self.doc_index.index(docs)
            LOG.info('Done adding the final batch documents.')

    def create_binary_format(self, concepts):
        indices = np.array(self.concept_tokenizer.encode(concepts)).flatten()
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
            LOG.warning(f'The index is empty at {self.index_folder}, please build the index first!')
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

        concept_embeddings = self.create_binary_format(concept_ids)

        query = (
            self.doc_index.build_query()  # get empty query object
            .filter(filter_query={'$and': [
                {'year': {'$gte': year - year_std}},
                {'year': {'$lte': year + year_std}},
                {'age': {'$gte': age - age_std}},
                {'age': {'$lte': age + age_std}},
                {'gender': {'$eq': gender}},
                {'race': {'$eq': race}}]
            })  # pre-filtering
            .find(query=concept_embeddings, search_field='concept_embeddings')  # add vector similarity search
            .build()
        )

        # execute the combined query and return the results
        results = self.doc_index.execute_query(query, limit=limit)
        return [vars(_) for _ in results.documents]

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
