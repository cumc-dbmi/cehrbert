import random
from abc import ABC
from enum import Enum

import networkx as nx
import pandas as pd


class SimilarityType(Enum):
    SEMANTIC_SIMILARITY = "semantic_similarity"
    MICA_INFORMATION_CONTENT = "mica_information_content"
    LIN_MEASURE = "lin_measure"
    JIANG_MEASURE = "jiang_measure"
    INFORMATION_COEFFICIENT = "information_coefficient"
    RELEVANCE_MEASURE = "relevance_measure"
    GRAPH_IC_MEASURE = "graph_ic_measure"
    NONE = "none"


class GraphSampler(ABC):
    # TODO: Weighted similarity to increase the sampling the local neighbors using logit
    def __init__(self, concept_similarity_path: str, concept_similarity_type: str):
        self._concept_similarity_type = concept_similarity_type
        self._concept_dict, self._similarity_dict = self._init_similarity(
            concept_similarity_type=concept_similarity_type,
            concept_similarity_path=concept_similarity_path,
        )

    def _is_sampling_enabled(self):
        """
        Check whether the graph sampling is enabled.

        :return:
        """
        return self._concept_similarity_type != SimilarityType.NONE.value

    def _init_similarity(self, concept_similarity_type: str, concept_similarity_path: str):
        concept_dict = {}
        similarity_dict = {}

        # Check whether we want to use similarity type
        if self._is_sampling_enabled():
            similarity_table = pd.read_parquet(concept_similarity_path)
            graph = nx.from_pandas_edgelist(
                similarity_table,
                source="concept_id_1",
                target="concept_id_2",
                edge_attr=concept_similarity_type,
            )

            for source in graph.nodes():
                # Convert target and source to str types because tokenizer expects string type
                concept_list, similarity_list = zip(
                    *[(str(target), val[concept_similarity_type]) for _, target, val in graph.edges(source, data=True)]
                )
                # Convert all concept_ids to string type
                concept_dict[str(source)] = list(map(str, concept_list))
                similarity_dict[str(source)] = similarity_list

        return concept_dict, similarity_dict

    def sample_graph(self, concept_id):
        # sample the concept from the probability distribution in the graph and generate one
        if self._is_sampling_enabled() and concept_id in self._concept_dict:
            # This actually returns a list
            random_choice = random.choices(
                population=self._concept_dict[concept_id],
                weights=self._similarity_dict[concept_id],
                k=1,
            )
            # In case the choice is none, we return itself as the default value
            return next(iter(random_choice), concept_id)

        # If the concept is an orphan, simply return it
        return concept_id
