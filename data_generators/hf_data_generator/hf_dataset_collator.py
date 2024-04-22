from typing import Any, Tuple
from models.hf_models.tokenization_hf_cehrbert import CehrBertTokenizer
from torch.nn.utils.rnn import pad_sequence
import torch


class CehrBertDataCollator:
    def __init__(
            self, tokenizer: CehrBertTokenizer,
            max_length: int,
            mlm_probability: float = 0.15,
            is_pretraining: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.is_pretraining = is_pretraining

    @staticmethod
    def _convert_to_tensor(features: Any) -> torch.Tensor:
        if isinstance(features, torch.Tensor):
            return features
        else:
            return torch.tensor(features)

    def __call__(self, examples):
        batch = {}
        batch_size = len(examples)

        # Assume that each example in the batch is a dictionary with 'input_ids' and 'attention_mask'
        batch_input_ids = [self._convert_to_tensor(example['input_ids']) for example in examples]
        batch_attention_mask = [torch.ones_like(self._convert_to_tensor(example['input_ids']), dtype=torch.float) for
                                example in examples]
        batch_ages = [self._convert_to_tensor(example['ages']) for example in examples]
        batch_dates = [self._convert_to_tensor(example['dates']) for example in examples]
        batch_visit_concept_orders = [self._convert_to_tensor(example['visit_concept_orders']) for example in examples]
        batch_concept_values = [self._convert_to_tensor(example['concept_values']) for example in examples]
        batch_concept_value_masks = [self._convert_to_tensor(example['concept_value_masks']) for example in examples]
        batch_visit_segments = [self._convert_to_tensor(example['visit_segments']) for example in examples]

        # Pad sequences to the max length in the batch
        batch['input_ids'] = pad_sequence(
            batch_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_index
        )
        batch['attention_mask'] = pad_sequence(
            batch_attention_mask,
            batch_first=True,
            padding_value=0.
        )
        batch['ages'] = pad_sequence(
            batch_ages,
            batch_first=True,
            padding_value=0
        )
        batch['dates'] = pad_sequence(
            batch_dates,
            batch_first=True,
            padding_value=0
        )
        batch['visit_concept_orders'] = pad_sequence(
            batch_visit_concept_orders,
            batch_first=True,
            padding_value=self.max_length - 1
        )
        batch['concept_values'] = pad_sequence(
            batch_concept_values,
            batch_first=True,
            padding_value=0.
        )
        batch['concept_value_masks'] = pad_sequence(
            batch_concept_value_masks,
            batch_first=True,
            padding_value=0.
        )
        batch['visit_segments'] = pad_sequence(
            batch_visit_segments,
            batch_first=True,
            padding_value=0
        )
        # Prepend the CLS token and their associated values to the corresponding time series features
        batch['input_ids'] = torch.cat(
            [torch.full((batch_size, 1), self.tokenizer.cls_token_index), batch['input_ids']],
            dim=1
        )
        # The attention_mask is set to 1 to enable attention for the CLS token
        batch['attention_mask'] = torch.cat(
            [torch.full((batch_size, 1), 1.0), batch['attention_mask']],
            dim=1
        )
        # Set the age of the CLS token to the starting age
        batch['ages'] = torch.cat(
            [batch['ages'][:, 0:1], batch['ages']],
            dim=1
        )
        # Set the age of the CLS token to the starting date
        batch['dates'] = torch.cat(
            [batch['dates'][:, 0:1], batch['dates']],
            dim=1
        )
        # Set the visit_concept_order of the CLS token to the first visit_concept_order in the sequence subtract by 1
        visit_concept_orders_first = batch['visit_concept_orders'][:, 0:1] - 1
        visit_concept_orders_first = torch.maximum(
            visit_concept_orders_first,
            torch.zeros_like(visit_concept_orders_first)
        )
        batch['visit_concept_orders'] = torch.cat(
            [visit_concept_orders_first, batch['visit_concept_orders']],
            dim=1
        )
        # Set the concept_value of the CLS token to a default value -1.0.
        batch['concept_values'] = torch.cat(
            [torch.full((batch_size, 1), -1.), batch['concept_values']],
            dim=1
        )
        # Set the concept_value of the CLS token to a default value 0.0 indicating that
        # there is no value associated with this token
        batch['concept_value_masks'] = torch.cat(
            [torch.full((batch_size, 1), 0.), batch['concept_value_masks']],
            dim=1
        )
        # Set the visit_segments of the CLS token to a default value 0 because this doesn't belong to a visit
        batch['visit_segments'] = torch.cat(
            [torch.full((batch_size, 1), 0), batch['visit_segments']],
            dim=1
        )

        # This is the most crucial logic for generating the training labels
        if self.is_pretraining:
            # If the labels field is already provided, we will build the MLM labels off of that.
            # The labels value indicates the positions that are not allowed for MLM.
            # For example, the mlm_skip_values=1, this means this is a lab value and
            # we don't want to predict the tokens at this position
            if 'labels' in examples[0]:
                batch_labels = [self._convert_to_tensor(example['labels']) for example in examples]
                batch['labels'] = pad_sequence(
                    batch_labels,
                    batch_first=True,
                    padding_value=-100
                )
                batch['labels'] = torch.cat(
                    [torch.full((batch_size, 1), -100), batch['labels']],
                    dim=1
                )
            else:
                # If the labels is not already provided, we assume all the tokens are subject to
                # the MLM and simply clone the input_ids
                batch['labels'] = batch['input_ids'].clone()

            batch['input_ids'], batch['labels'] = self.torch_mask_tokens(batch['input_ids'], batch['labels'])

        if 'age_at_index' in examples[0]:
            batch['age_at_index'] = torch.cat(
                [self._convert_to_tensor(example['age_at_index']).reshape(-1, 1) for example in examples],
                dim=0
            ).to(torch.float)

        if 'classifier_label' in examples[0]:
            batch['classifier_label'] = torch.cat(
                [self._convert_to_tensor(example['classifier_label']).reshape(-1, 1) for example in examples],
                dim=0
            ).to(torch.float)

        return batch

    def torch_mask_tokens(self, inputs: torch.Tensor, labels: torch.Tensor) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
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
