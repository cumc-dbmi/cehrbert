from typing import Type, List, Dict

from docarray import BaseDoc
from docarray.index import WeaviateDocumentIndex
from docarray.typing import NdArray
from pydantic import Field

from analyses.gpt.privacy.patient_index.base_indexer import PatientDataIndex, LOG


class PatientDataWeaviateDocumentIndex(PatientDataIndex):

    def __init__(
            self,
            server_name: str,
            index_name: str,
            *args,
            **kwargs
    ):
        self.index_name = index_name
        self.server_name = server_name
        super(PatientDataWeaviateDocumentIndex, self).__init__(*args, **kwargs)
        LOG.info(
            f'\tserver_name: {server_name}\n'
            f'\tindex_name: {index_name}\n'
        )

    def create_index(self) -> WeaviateDocumentIndex[BaseDoc]:
        dbconfig = WeaviateDocumentIndex.DBConfig(
            host=self.server_name
        )  # Rep
        return WeaviateDocumentIndex[self.doc_class](db_config=dbconfig, index_name=self.index_name)

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
            concept_embeddings: NdArray[vocab_size] = Field(is_embedding=True)
            sensitive_attributes: str
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

        query = {
            "operator": "And",
            "operands": [{
                "path": ["gender"],
                "operator": "Equal",
                "valueText": gender
            }, {
                "path": ["race"],
                "operator": "Equal",
                "valueText": race,
            }, {
                "path": ["age"],
                "operator": "LessThan",
                "valueInt": age + age_std,
            }, {
                "path": ["age"],
                "operator": "GreaterThan",
                "valueInt": age - age_std
            }, {
                "path": ["year"],
                "operator": "LessThan",
                "valueInt": year + year_std,
            }, {
                "path": ["year"],
                "operator": "GreaterThan",
                "valueInt": year - year_std
            }]
        }

        query = (
            self.doc_index.build_query()  # get empty query object
            .filter(where_filter=query)  # pre-filtering
            .find(concept_embeddings)  # add vector similarity search
            .build()
        )

        # execute the combined query and return the results
        results = self.doc_index.execute_query(query, limit=limit)
        dicts = [vars(_) for _ in results]
        for d in dicts:
            d['sensitive_attributes'] = d['sensitive_attributes'].split(',')
        return dicts
