import os
import logging
from typing import Union, List, Generator
from tqdm import tqdm
from enum import Enum

from whoosh import index
from whoosh.fields import Schema, TEXT, NUMERIC, ID
from whoosh.analysis import RegexTokenizer

from whoosh.qparser import QueryParser, RangePlugin
from whoosh.query import And
from whoosh.searching import Hit

from dask.dataframe import DataFrame as dd_dataframe
from pandas import DataFrame as pd_dataframe

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("whoosh_index")


class ConceptLogicOperator(Enum):
    OR = 'OR'
    AND = 'AND'


class PatientDataIndex:
    def __init__(
            self,
            index_folder: str,
            rebuilt: bool = False
    ):
        self.index_folder = index_folder
        self.rebuilt = rebuilt

        LOG.info(
            f'The PatientDataIndex parameters\n:'
            f'\tindex_folder: {index_folder}\n'
            f'\trebuilt: {rebuilt}\n'
        )

        # Get the index for search
        self.search_ix = index.open_dir(self.index_folder) if index.exists_in(self.index_folder) else None

    def build_index(
            self,
            dataset: Union[pd_dataframe, dd_dataframe]
    ):
        LOG.info('Started adding documents to the index.')

        # Create a directory to store the index
        if not os.path.exists(self.index_folder):
            LOG.info(f'The index folder {self.index_folder} does not exist, creating it')
            os.mkdir(self.index_folder)

        # build the index
        if index.exists_in(self.index_folder) and not self.rebuilt:
            LOG.warning(
                f'The index already exists in {self.index_folder}. '
                f'If you want to rebuild the index, you could set rebuilt=True'
            )
            return
        else:
            # Create the index schema
            index.create_in(self.index_folder, self.get_index_schema())

        # Get the index
        ix = index.open_dir(self.index_folder)
        writer = ix.writer()

        for t in tqdm(dataset.itertuples(), total=len(dataset)):
            if self.validate_demographics(t.concept_ids):
                person_id = str(t.person_id)
                year, age, gender, race = self.get_demographics(t.concept_ids)
                # Concatenate all the concept ids using whitespace
                concept_ids_str = ' '.join([_ for _ in t.concept_ids[4:] if str.isnumeric(_)])

                # Add a document to the index at a time
                writer.add_document(
                    person_id=person_id,
                    year=year,
                    age=age,
                    gender=gender,
                    race=race,
                    concepts=concept_ids_str
                )

        LOG.info('Done adding documents.')
        LOG.info('Started committing the changes to the index.')
        # commit the changes
        writer.commit()
        LOG.info('Done committing.')

    def search(
            self,
            patient_seq: List[str],
            year_std: int = 1,
            age_std: int = 1,
            concepts_logic_operator: ConceptLogicOperator = ConceptLogicOperator.OR,
            limit: int = 1
    ) -> List[Hit]:

        if not self.search_ix:
            LOG.warning(f'The index is empty at {self.index_folder}, please build the index first!')
            return

        if not self.validate_demographics(patient_seq):
            LOG.warning(f'The first four tokens {patient_seq[0:4]} do not contain valid demographic information!')
            return

        year, age, gender, race = self.get_demographics(patient_seq)
        concept_ids = [_ for _ in patient_seq[4:] if str.isnumeric(_)]
        return [_ for _ in self.find_patients(
            year, age, gender, race, concept_ids, year_std, age_std, concepts_logic_operator, limit
        )]

    def find_patients(
            self,
            year: int,
            age: int,
            gender: str,
            race: str,
            concept_ids: List[str],
            year_std: int,
            age_std: int,
            concepts_logic_operator: ConceptLogicOperator,
            limit: int
    ) -> Generator[Hit, None, None]:

        # Create a query parser for the range field
        year_parser = QueryParser('year', self.search_ix.schema)
        year_parser.add_plugin(RangePlugin())
        age_parser = QueryParser('age', self.search_ix.schema)
        age_parser.add_plugin(RangePlugin())

        # Create a query parser for the text field
        gender_parser = QueryParser('gender', self.search_ix.schema)
        race_parser = QueryParser('race', self.search_ix.schema)
        concepts_parser = QueryParser('concepts', self.search_ix.schema)

        # Define your text and range queries
        year_query = year_parser.parse(f'[{year - year_std} to {year + year_std}]')
        age_query = age_parser.parse(f'[{age - age_std} to {age + age_std}]')
        gender_query = gender_parser.parse(gender)
        race_query = race_parser.parse(race)
        concepts_query = concepts_parser.parse(f' {concepts_logic_operator.value} '.join(concept_ids))

        combined_query = And([year_query, age_query, race_query, gender_query, concepts_query])
        # Searching
        with self.search_ix.searcher() as searcher:
            results = searcher.search(combined_query, limit=limit)
            for result in results:
                yield result

    @staticmethod
    def get_index_schema():
        # Basic tokenizer without further processing
        basic_analyzer = RegexTokenizer()
        # Define the schema for your index
        schema = Schema(
            person_id=ID(stored=True),
            year=NUMERIC(stored=True),
            age=NUMERIC(stored=True),
            race=TEXT(stored=True, analyzer=basic_analyzer),
            gender=TEXT(stored=True, analyzer=basic_analyzer),
            concepts=TEXT(stored=True)
        )
        return schema

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
