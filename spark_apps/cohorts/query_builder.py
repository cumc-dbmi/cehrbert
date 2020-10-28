from abc import ABC
from typing import List, NamedTuple
import logging


class QuerySpec(NamedTuple):
    query_template: str
    parameters: dict
    table_name: str

    def __str__(self):
        return (f'table={self.table_name}\n'
                f'query={self.query_template.format(**self.parameters)}\n')


class AncestorTableSpec(NamedTuple):
    ancestor_concept_ids: List[int]
    table_name: str
    is_standard: bool

    def __str__(self):
        return (f'table_name={self.table_name}\n'
                f'ancestor_concept_ids={self.ancestor_concept_ids}\n'
                f'is_standard={self.is_standard}\n')


class QueryBuilder(ABC):

    def __init__(self,
                 cohort_name: str,
                 dependency_list: List[str],
                 query: QuerySpec,
                 dependency_queries: List[QuerySpec] = None,
                 post_queries: List[QuerySpec] = None,
                 ancestor_table_specs: List[AncestorTableSpec] = None):
        """

        :param cohort_name:
        :param query:
        :param dependency_queries:
        :param post_queries:
        :param dependency_list:
        :param ancestor_table_specs:
        """
        self._cohort_name = cohort_name
        self._query = query
        self._dependency_queries = dependency_queries
        self._post_queries = post_queries
        self._dependency_list = dependency_list
        self._ancestor_table_specs = ancestor_table_specs

        self.get_logger().info(f'cohort_name: {cohort_name}\n'
                               f'post_queries: {post_queries}\n'
                               f'dependency_queries: {dependency_queries}\n'
                               f'dependency_list: {dependency_list}\n'
                               f'ancestor_table_specs: {ancestor_table_specs}\n'
                               f'query: {query}\n')

    def get_dependency_queries(self):
        """
        Instantiate table dependencies in spark for
        :return:
        """
        return self._dependency_queries

    def get_query(self):
        """
        Create a query that can be executed by spark.sql
        :return:
        """
        return self._query

    def get_post_process_queries(self):
        """
        Get a list of post process queries to process the cohort
        :return:
        """
        return self._post_queries

    def get_dependency_list(self):
        """
        Get a list of tables that are required for this cohort
        :return:
        """
        return self._dependency_list

    def get_cohort_name(self):
        return self._cohort_name

    def get_ancestor_table_specs(self):
        """
        Create the descendant table for the provided ancestor_table_specs
        :return:
        """
        return self._ancestor_table_specs

    def __str__(self):
        return f'{str(self.__class__.__name__)} for {self.get_cohort_name()}'

    @classmethod
    def get_logger(cls):
        return logging.getLogger(cls.__name__)
