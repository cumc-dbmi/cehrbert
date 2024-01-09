from tqdm import tqdm
from typing import Type, List, Dict, Sequence

from docarray import BaseDoc
from docarray.index.backends.weaviate import WeaviateDocumentIndex, DOCUMENTID
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
            concept_ids: str
            sensitive_attributes: str
            num_of_visits: int
            num_of_concepts: int

        return PatientDocument

    def _get_batch_with_cursor(self, cursor):
        # First prepare the query to run through data
        query = (
            self.doc_index._client.query.get(
                self.doc_index.index_name,
                ["person_id"]
            )
            .with_additional(["id"])
            .with_limit(self.batch_size)
        )
        # Fetch the next set of results
        if cursor is not None:
            result = query.with_after(cursor).do()
        # Fetch the first set of results
        else:
            result = query.do()
        return result["data"]["Get"][self.doc_index.index_name]

    def _del_items(self, doc_ids: Sequence[str]):
        has_matches = True
        if doc_ids:
            where_filter = {
                "path": [DOCUMENTID],
                "operator": "ContainsAny",
                "valueTextArray": doc_ids
            }
            # do a loop because there is a limit to how many objects can be deleted at
            # in a single query
            # see: https://weaviate.io/developers/weaviate/api/rest/batch#maximum-number-of-deletes-per-query
            while has_matches:
                results = self.doc_index._client.batch.delete_objects(
                    class_name='PatientDocument',
                    where=where_filter,
                )
                has_matches = results["results"]["matches"]

    def delete_index(self):
        pbar = tqdm()
        cursor = None
        while True:
            # Get the next batch of objects
            next_batch = self._get_batch_with_cursor(cursor)
            # Break the loop if empty – we are done
            if len(next_batch) == 0:
                break
            # Here is your next batch of objects
            self._del_items([d[DOCUMENTID] for d in next_batch])
            # Move the cursor to the last returned uuid
            cursor = next_batch[-1]["_additional"]["id"]
            # Update the progress bard
            pbar.update(1)

    def get_all_person_ids(self) -> List[int]:

        person_ids = []
        pbar = tqdm()
        cursor = None
        while True:
            # Get the next batch of objects
            next_batch = self._get_batch_with_cursor(cursor)
            # Break the loop if empty – we are done
            if len(next_batch) == 0:
                break
            # Here is your next batch of objects
            person_ids.extend([int(d['person_id']) for d in next_batch])
            # Move the cursor to the last returned uuid
            cursor = next_batch[-1]["_additional"]["id"]
            # Update the progress bard
            pbar.update(1)

        return person_ids

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
            .limit(limit)
            .build()
        )

        # execute the combined query and return the results
        results = self.doc_index.execute_query(query)
        dicts = [vars(_) for _ in results]
        for d in dicts:
            if d['sensitive_attributes']:
                d['sensitive_attributes'] = d['sensitive_attributes'].split(',')
            if d['concept_ids']:
                d['concept_ids'] = d['concept_ids'].split(',')
        return dicts
