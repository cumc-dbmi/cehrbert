from typing import Type, List, Dict

from docarray import BaseDoc
from docarray.index import HnswDocumentIndex
from docarray.typing import NdArray

from analyses.gpt.privacy.patient_index.base_indexer import PatientDataIndex


class PatientDataHnswDocumentIndex(PatientDataIndex):

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
            sensitive_attributes: List[str]
            num_of_visits: int
            num_of_concepts: int

        return PatientDocument

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
