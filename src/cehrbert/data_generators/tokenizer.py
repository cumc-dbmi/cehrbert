from typing import Optional, Sequence, Union

from cehrbert_data.const.common import UNKNOWN_CONCEPT
from dask.dataframe import Series as dd_series
from pandas import Series as df_series
from tensorflow.keras.preprocessing.text import Tokenizer


class ConceptTokenizer:
    unused_token = "[UNUSED]"
    mask_token = "[MASK]"
    att_mask_token = "[ATT_MASK]"
    cls_token = "[CLS]"
    start_token = "[START]"
    end_token = "[END]"
    visit_start_token = "VS"
    visit_end_token = "VE"

    def __init__(self, special_tokens: Optional[Sequence[str]] = None, oov_token="-1"):
        self.special_tokens = special_tokens
        self.tokenizer = Tokenizer(oov_token=oov_token, filters="", lower=False)

        self.tokenizer.fit_on_texts([self.mask_token])
        self.tokenizer.fit_on_texts([self.att_mask_token])
        self.tokenizer.fit_on_texts([self.unused_token])
        self.tokenizer.fit_on_texts([self.cls_token])
        self.tokenizer.fit_on_texts([self.start_token])
        self.tokenizer.fit_on_texts([self.end_token])
        self.tokenizer.fit_on_texts([UNKNOWN_CONCEPT])
        self.tokenizer.fit_on_texts([self.visit_start_token])
        self.tokenizer.fit_on_texts([self.visit_end_token])

        if self.special_tokens is not None:
            self.tokenizer.fit_on_texts(self.special_tokens)

    def fit_on_concept_sequences(self, concept_sequences: Union[df_series, dd_series]):
        if isinstance(concept_sequences, df_series):
            self.tokenizer.fit_on_texts(concept_sequences.apply(list))
        else:
            self.tokenizer.fit_on_texts(concept_sequences.apply(list, meta="iterable"))

    def encode(self, concept_sequences, is_generator=False):
        return (
            self.tokenizer.texts_to_sequences_generator(concept_sequences)
            if is_generator
            else self.tokenizer.texts_to_sequences(concept_sequences)
        )

    def decode(self, concept_sequence_token_ids):
        return self.tokenizer.sequences_to_texts(concept_sequence_token_ids)

    def get_all_tokens(self):
        return set(self.tokenizer.word_index.keys())

    def get_all_token_indexes(self):
        all_keys = set(self.tokenizer.index_word.keys())

        if self.tokenizer.oov_token is not None:
            all_keys.remove(self.tokenizer.word_index[self.tokenizer.oov_token])

        if self.special_tokens is not None:
            excluded = set([self.tokenizer.word_index[special_token] for special_token in self.special_tokens])
            all_keys = all_keys - excluded
        return all_keys

    def get_token_by_index(self, index):
        if index in self.tokenizer.index_word:
            return self.tokenizer.index_word[index]
        raise RuntimeError(f"{index} is not a valid index in tokenizer")

    def get_first_token_index(self):
        return min(self.get_all_token_indexes())

    def get_last_token_index(self):
        return max(self.get_all_token_indexes())

    def get_vocab_size(self):
        # + 1 because oov_token takes the index 0
        return len(self.tokenizer.index_word) + 1

    def get_start_token_id(self):
        start_token_id = self.encode([self.start_token])
        while isinstance(start_token_id, list):
            start_token_id = start_token_id[0]
        return start_token_id

    def get_end_token_id(self):
        end_token_id = self.encode([self.end_token])
        while isinstance(end_token_id, list):
            end_token_id = end_token_id[0]
        return end_token_id

    def get_unused_token_id(self):
        unused_token_id = self.encode([self.unused_token])
        while isinstance(unused_token_id, list):
            unused_token_id = unused_token_id[0]
        return unused_token_id

    def get_unused_token(self):
        return self.unused_token

    def get_mask_token_id(self):
        mask_token_id = self.encode([self.mask_token])
        while isinstance(mask_token_id, list):
            mask_token_id = mask_token_id[0]
        return mask_token_id

    def get_mask_token(self):
        return self.mask_token

    def get_att_mask_token_id(self):
        att_mask_token_id = self.encode([self.att_mask_token])
        while isinstance(att_mask_token_id, list):
            att_mask_token_id = att_mask_token_id[0]
        return att_mask_token_id

    def get_att_mask_token(self):
        return self.att_mask_token

    def get_visit_start_token_id(self):
        visit_start_token_id = self.encode([self.visit_start_token])
        while isinstance(visit_start_token_id, list):
            visit_start_token_id = visit_start_token_id[0]
        return visit_start_token_id

    def get_visit_end_token_id(self):
        visit_end_token_id = self.encode([self.visit_end_token])
        while isinstance(visit_end_token_id, list):
            visit_end_token_id = visit_end_token_id[0]
        return visit_end_token_id
