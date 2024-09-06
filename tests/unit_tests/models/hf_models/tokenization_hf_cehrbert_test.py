import unittest

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from cehrbert.models.hf_models.tokenization_hf_cehrbert import (
    CLS_TOKEN,
    MASK_TOKEN,
    OUT_OF_VOCABULARY_TOKEN,
    PAD_TOKEN,
    UNUSED_TOKEN,
    CehrBertTokenizer,
)


class TestCehrBertTokenizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Setup with a small example vocabulary and concept mapping
        vocab = {
            PAD_TOKEN: 0,
            MASK_TOKEN: 1,
            OUT_OF_VOCABULARY_TOKEN: 2,
            UNUSED_TOKEN: 3,
            CLS_TOKEN: 4,
            "hello": 5,
            "world": 6,
        }
        tokenizer = Tokenizer(WordLevel(unk_token=OUT_OF_VOCABULARY_TOKEN, vocab=vocab))
        tokenizer.pre_tokenizer = Whitespace()

        concept_mapping = {"hello": "Hello", "world": "World"}
        cls.tokenizer = CehrBertTokenizer(tokenizer, lab_stats=[], concept_name_mapping=concept_mapping)

    def test_vocab_size(self):
        # Test the vocabulary size
        self.assertEqual(self.tokenizer.vocab_size, 7)

    def test_encode(self):
        # Test the encoding method
        encoded = self.tokenizer.encode(["hello", "world"])
        self.assertEqual(encoded, [5, 6])

    def test_decode(self):
        # Test the decoding method
        decoded = self.tokenizer.decode([5, 6])
        self.assertEqual(decoded, ["hello", "world"])

    def test_convert_tokens_to_string(self):
        # Test converting tokens to a single string
        result = self.tokenizer.convert_tokens_to_string(["hello", "world"])
        self.assertEqual(result, "Hello World")

    def test_oov_token(self):
        # Test the encoding of an out-of-vocabulary token
        encoded = self.tokenizer.encode(["nonexistent"])
        self.assertEqual(encoded, [self.tokenizer.oov_token_index])

    def test_convert_id_to_token_oov(self):
        # Test decoding an out-of-vocabulary token ID
        decoded = self.tokenizer.convert_id_to_token(99)  # Assuming 99 is not in the index
        self.assertEqual(decoded, OUT_OF_VOCABULARY_TOKEN)


if __name__ == "__main__":
    unittest.main()
