import collections
import random
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from cehrbert.data_generators.hf_data_generator.hf_dataset_mapping import TruncationType
from cehrbert.models.hf_models.tokenization_hf_cehrbert import CehrBertTokenizer


class CehrBertDataCollator:
    def __init__(
        self,
        tokenizer: CehrBertTokenizer,
        max_length: int,
        mlm_probability: float = 0.15,
        is_pretraining: bool = True,
        truncate_type: TruncationType = TruncationType.RANDOM_RIGHT_TRUNCATION,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.is_pretraining = is_pretraining
        # We only enable random truncations during pretraining
        # We always take the most recent history during finetuning indicated by TruncationType.TAIL
        self.truncate_type = truncate_type if is_pretraining else TruncationType.TAIL
        # Pre-compute these so we can use them later on
        # We used VS for the historical data, currently, we use the new [VS] for the newer data
        # so we need to check both cases.
        self.vs_token_id = tokenizer.convert_token_to_id("VS")
        if self.vs_token_id == tokenizer.oov_token_index:
            self.vs_token_id = tokenizer.convert_token_to_id("[VS]")
        self.ve_token_id = tokenizer.convert_token_to_id("VE")
        if self.ve_token_id == tokenizer.oov_token_index:
            self.ve_token_id = tokenizer.convert_token_to_id("[VE]")

    @staticmethod
    def _convert_to_tensor(features: Any) -> torch.Tensor:
        if isinstance(features, torch.Tensor):
            return features
        else:
            return torch.tensor(features)

    def __call__(self, examples):

        examples = [self.generate_start_end_index(_) for _ in examples]
        batch = {}
        batch_size = len(examples)

        # Assume that each example in the batch is a dictionary with 'input_ids' and 'attention_mask'
        batch_input_ids = [self._convert_to_tensor(example["input_ids"]) for example in examples]
        batch_attention_mask = [
            torch.ones_like(self._convert_to_tensor(example["input_ids"]), dtype=torch.float) for example in examples
        ]
        batch_ages = [self._convert_to_tensor(example["ages"]) for example in examples]
        batch_dates = [self._convert_to_tensor(example["dates"]) for example in examples]
        batch_visit_concept_orders = [self._convert_to_tensor(example["visit_concept_orders"]) for example in examples]
        batch_concept_values = [self._convert_to_tensor(example["concept_values"]) for example in examples]
        batch_concept_value_masks = [self._convert_to_tensor(example["concept_value_masks"]) for example in examples]
        batch_visit_segments = [self._convert_to_tensor(example["visit_segments"]) for example in examples]

        # Pad sequences to the max length in the batch
        batch["input_ids"] = pad_sequence(
            batch_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_index,
        )
        batch["attention_mask"] = pad_sequence(batch_attention_mask, batch_first=True, padding_value=0.0)
        batch["ages"] = pad_sequence(batch_ages, batch_first=True, padding_value=0)
        batch["dates"] = pad_sequence(batch_dates, batch_first=True, padding_value=0)
        batch["visit_concept_orders"] = pad_sequence(
            batch_visit_concept_orders,
            batch_first=True,
            padding_value=self.max_length - 1,
        )
        batch["concept_values"] = pad_sequence(batch_concept_values, batch_first=True, padding_value=0.0)
        batch["concept_value_masks"] = pad_sequence(batch_concept_value_masks, batch_first=True, padding_value=0.0)
        batch["visit_segments"] = pad_sequence(batch_visit_segments, batch_first=True, padding_value=0)

        # Prepend the CLS token and their associated values to the corresponding time series features
        batch["input_ids"] = torch.cat(
            [
                torch.full((batch_size, 1), self.tokenizer.cls_token_index),
                batch["input_ids"],
            ],
            dim=1,
        )
        # The attention_mask is set to 1 to enable attention for the CLS token
        batch["attention_mask"] = torch.cat([torch.full((batch_size, 1), 1.0), batch["attention_mask"]], dim=1)
        # Set the age of the CLS token to the starting age
        batch["ages"] = torch.cat([batch["ages"][:, 0:1], batch["ages"]], dim=1)
        # Set the age of the CLS token to the starting date
        batch["dates"] = torch.cat([batch["dates"][:, 0:1], batch["dates"]], dim=1)
        # Set the visit_concept_order of the CLS token to the first visit_concept_order in the sequence subtract by 1
        visit_concept_orders_first = batch["visit_concept_orders"][:, 0:1] - 1
        visit_concept_orders_first = torch.maximum(
            visit_concept_orders_first, torch.zeros_like(visit_concept_orders_first)
        )
        batch["visit_concept_orders"] = torch.cat([visit_concept_orders_first, batch["visit_concept_orders"]], dim=1)
        # Set the concept_value of the CLS token to a default value -1.0.
        batch["concept_values"] = torch.cat([torch.full((batch_size, 1), -1.0), batch["concept_values"]], dim=1)
        # Set the concept_value of the CLS token to a default value 0.0 indicating that
        # there is no value associated with this token
        batch["concept_value_masks"] = torch.cat(
            [torch.full((batch_size, 1), 0.0), batch["concept_value_masks"]], dim=1
        )
        # Set the visit_segments of the CLS token to a default value 0 because this doesn't belong to a visit
        batch["visit_segments"] = torch.cat([torch.full((batch_size, 1), 0), batch["visit_segments"]], dim=1)

        # This is the most crucial logic for generating the training labels
        if self.is_pretraining:

            batch_mlm_skip_values = [
                self._convert_to_tensor(example["mlm_skip_values"]).to(torch.bool) for example in examples
            ]
            batch["mlm_skip_values"] = pad_sequence(batch_mlm_skip_values, batch_first=True, padding_value=False)
            # Set the mlm_skip_values of the CLS token to a default value False
            batch["mlm_skip_values"] = torch.cat([torch.full((batch_size, 1), False), batch["mlm_skip_values"]], dim=1)

            # If the labels field is already provided, we will build the MLM labels off of that.
            # The labels value indicates the positions that are not allowed for MLM.
            # For example, the mlm_skip_values=1, this means this is a lab value and
            # we don't want to predict the tokens at this position
            if "labels" in examples[0]:
                batch_labels = [self._convert_to_tensor(example["labels"]) for example in examples]
                batch["labels"] = pad_sequence(batch_labels, batch_first=True, padding_value=-100)
                # Disable MLM for the CLS token
                batch["labels"] = torch.cat([torch.full((batch_size, 1), -100), batch["labels"]], dim=1)
            else:
                # If the labels is not already provided, we assume all the tokens are subject to
                # the MLM and simply clone the input_ids
                batch["labels"] = batch["input_ids"].clone()

            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(batch["input_ids"], batch["labels"])

        if "person_id" in examples[0]:
            batch["person_id"] = torch.cat(
                [self._convert_to_tensor(example["person_id"]).reshape(-1, 1) for example in examples],
                dim=0,
            ).to(torch.float)

        if "index_date" in examples[0]:
            batch["index_date"] = torch.cat(
                [self._convert_to_tensor(example["index_date"]).reshape(-1, 1) for example in examples],
                dim=0,
            ).to(torch.float32)

        if "age_at_index" in examples[0]:
            batch["age_at_index"] = torch.cat(
                [self._convert_to_tensor(example["age_at_index"]).reshape(-1, 1) for example in examples],
                dim=0,
            ).to(torch.float)

        if "classifier_label" in examples[0]:
            batch["classifier_label"] = torch.cat(
                [self._convert_to_tensor(example["classifier_label"]).reshape(-1, 1) for example in examples],
                dim=0,
            ).to(torch.float)

        return batch

    def torch_mask_tokens(self, inputs: torch.Tensor, labels: torch.Tensor) -> Tuple[Any, Any]:
        """Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original."""
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        pad_token_mask = inputs == self.tokenizer.pad_token_index

        probability_matrix.masked_fill_(pad_token_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_index

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.tokenizer.vocab_size, labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def generate_start_end_index(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapted from https://github.com/OHDSI/Apollo/blob/main/data_loading/data_transformer.py.

        Adding the start and end indices to extract a portion of the patient sequence
        """
        seq_length = len(record["input_ids"])
        new_max_length = self.max_length - 1  # Subtract one for the [CLS] token

        # Return the record directly if the actual sequence length is less than the max sequence
        if seq_length <= new_max_length:
            return record

        # If it is random truncate, we don't care if we slice at the start or end of a visit
        if self.truncate_type == TruncationType.RANDOM_TRUNCATION:
            start_index = random.randint(0, seq_length - new_max_length)
            end_index = min(seq_length, start_index + new_max_length)
        elif self.truncate_type in (
            TruncationType.RANDOM_RIGHT_TRUNCATION,
            TruncationType.RANDOM_COMPLETE,
        ):
            # We randomly pick a [VS] token
            starting_points = []
            for i in range(seq_length - new_max_length):
                current_token = record["input_ids"][i]
                if current_token == self.vs_token_id:
                    starting_points.append(i)

            assert len(starting_points) > 0, f"{record['input_ids'][:seq_length - new_max_length]}"
            start_index = random.choice(starting_points)
            end_index = min(start_index + new_max_length, seq_length)

            # We randomly backtrack to a [VE] token so the sample is complete
            if self.truncate_type == TruncationType.RANDOM_COMPLETE:
                for i in reversed(list(range(start_index + 1, end_index))):
                    current_token = record["input_ids"][i]
                    if current_token == self.ve_token_id:
                        end_index = i
                        break
        else:
            start_index = max(0, seq_length - new_max_length)
            end_index = seq_length
            for i in range(start_index, seq_length):
                current_token = record["input_ids"][i]
                if current_token == self.vs_token_id:
                    start_index = i
                    break

        new_record = collections.OrderedDict()
        for k, v in record.items():
            if isinstance(v, list) or isinstance(v, np.ndarray) or (isinstance(v, torch.Tensor) and v.dim() > 0):
                if len(v) == seq_length:
                    new_record[k] = v[start_index:end_index]
            else:
                new_record[k] = v
        return new_record
