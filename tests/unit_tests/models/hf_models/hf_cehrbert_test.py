import unittest

import torch

from cehrbert.models.hf_models.config import CehrBertConfig
from cehrbert.models.hf_models.hf_cehrbert import CehrBertForPreTraining


class TestCehrBert(unittest.TestCase):

    def setUp(self):
        # Setup the configuration and model for testing
        self.config = CehrBertConfig(output_attentions=True)
        self.model = CehrBertForPreTraining(self.config)

    def test_positional_encoding_layer_output_shape(self):
        # Test the shape of the output from the PositionalEncodingLayer
        layer = self.model.bert.cehr_bert_embeddings.positional_embedding_layer
        seq_length = 10  # example sequence length
        embedding_size = self.config.n_time_embd
        output = layer(torch.randint(0, seq_length, (1, seq_length)))
        self.assertEqual(output.shape, (1, seq_length, embedding_size))

    def test_positional_encoding_layer_output_large_positions(self):
        # Test the shape of the output from the PositionalEncodingLayer
        layer = self.model.bert.cehr_bert_embeddings.positional_embedding_layer
        embedding_size = self.config.n_time_embd
        visit_concept_orders = torch.arange(10, 15).unsqueeze(0).repeat(2, 1)
        visit_concept_orders = torch.cat([visit_concept_orders, torch.full((2, 1), 5)], dim=-1)
        output = layer(visit_concept_orders)
        self.assertEqual(output.shape, (2, 6, embedding_size))

    def test_time_embedding_layer_output_shape(self):
        # Test the output shape of the TimeEmbeddingLayer
        layer = self.model.bert.cehr_bert_embeddings.time_embedding_layer
        seq_length = 10  # example sequence length
        time_stamps = torch.rand(1, seq_length)
        output = layer(time_stamps)
        self.assertEqual(output.shape, (1, seq_length, self.config.n_time_embd))

    def test_concept_value_transformation_layer_output_shape(self):
        # Test the output shape of the ConceptValueTransformationLayer
        layer = self.model.bert.cehr_bert_embeddings.concept_value_transformation_layer
        seq_length = 10  # example sequence length
        embedding_size = self.config.hidden_size
        concept_embeddings = torch.rand(1, seq_length, embedding_size)
        concept_values = torch.rand(1, seq_length)
        concept_value_masks = torch.ones(1, seq_length)
        output = layer(concept_embeddings, concept_values, concept_value_masks)
        self.assertEqual(output.shape, (1, seq_length, embedding_size))

    def test_model_output(self):
        # Test the output of the CehrBert model
        input_ids = torch.randint(0, self.config.vocab_size, (1, 10))
        attention_mask = torch.ones(1, 10)
        ages = torch.randint(0, 100, (1, 10))
        dates = torch.randint(3000, 3500, (1, 10))
        visit_concept_orders = torch.randint(0, 10, (1, 10))
        concept_values = torch.rand(1, 10)
        concept_value_masks = torch.randint(0, 2, (1, 10))
        mlm_skip_values = torch.randint(0, 2, (1, 10)).to(torch.bool)
        visit_segments = torch.randint(0, self.config.n_visit_segments, (1, 10))

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ages=ages,
            dates=dates,
            visit_concept_orders=visit_concept_orders,
            concept_values=concept_values,
            concept_value_masks=concept_value_masks,
            visit_segments=visit_segments,
            labels=input_ids,
            mlm_skip_values=mlm_skip_values,
        )

        self.assertTrue(hasattr(output, "loss"))
        self.assertTrue(hasattr(output, "last_hidden_state"))
        self.assertTrue(hasattr(output, "attentions"))
        self.assertTrue(hasattr(output, "prediction_logits"))
        self.assertTrue(hasattr(output, "pooler_output"))

        self.assertEqual(output.prediction_logits.shape, torch.Size([1, 10, self.config.vocab_size]))
        self.assertEqual(output.pooler_output.shape, torch.Size([1, 128]))
        self.assertEqual(output.last_hidden_state.shape, torch.Size([1, 10, self.config.hidden_size]))
        self.assertEqual(len(output.attentions), self.config.num_hidden_layers)
        self.assertEqual(
            output.attentions[0].shape,
            torch.Size([1, self.config.num_attention_heads, 10, 10]),
        )


if __name__ == "__main__":
    unittest.main()
