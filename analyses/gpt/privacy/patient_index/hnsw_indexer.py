import os
import shutil
from tqdm import tqdm
from typing import Type, List, Dict

from docarray import BaseDoc
from docarray.index import HnswDocumentIndex
from docarray.typing import NdArray
from docarray.proto import DocProto

from analyses.gpt.privacy.patient_index.base_indexer import PatientDataIndex, LOG


class PatientDataHnswDocumentIndex(PatientDataIndex):

    def get_all_person_ids(self) -> List[int]:
        person_ids = []
        self.doc_index._sqlite_cursor.execute('SELECT * FROM docs')
        for row in tqdm(self.doc_index._sqlite_cursor):
            pb = DocProto.FromString(
                row[1]
            )
            person_ids.append(int(pb.data.get('person_id').text))
        return person_ids

    def create_index(self) -> HnswDocumentIndex[BaseDoc]:
        return HnswDocumentIndex[self.doc_class](work_dir=self.index_folder)

    def create_doc_class(self) -> Type[BaseDoc]:
        """
        Factory function to create a class with a dynamic embedding size

        :return:
        """
        vocab_size = self.concept_tokenizer.get_vocab_size()

        class PatientDocument(BaseDoc):
            id: str
            person_id: str
            year: int
            age: int
            race: str
            gender: str
            concept_embeddings: NdArray[vocab_size]
            concept_ids: List[str]
            sensitive_attributes: List[str]
            num_of_visits: int
            num_of_concepts: int

        return PatientDocument

    def delete_index(self):
        try:
            shutil.rmtree(self.index_folder)
            os.mkdir(self.index_folder)
            LOG.info(f"The directory {self.index_folder} and all its contents have been removed")
            self.doc_index = HnswDocumentIndex[self.doc_class](work_dir=self.index_folder)
        except OSError as error:
            LOG.error(f"Error: {error}")
            LOG.error(f"Could not remove the directory {self.index_folder}")
            raise error

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
            .find(query=concept_embeddings, search_field='concept_embeddings', limit=limit)
            .build()
        )

        # execute the combined query and return the results
        results = self.doc_index.execute_query(query)
        return [vars(_) for _ in results.documents]
