from models.hf_models.tokenization_hf_cehrbert import CehrBertTokenizer
from torch.nn.utils.rnn import pad_sequence
import torch


class CehrBertDataCollator:
    def __init__(self, tokenizer: CehrBertTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        batch = {}

        has_label = 'labels' in examples[0]
        # Assume that each example in the batch is a dictionary with 'input_ids' and 'attention_mask'
        batch_input_ids = [example['input_ids'] for example in examples]
        batch_attention_mask = [torch.ones_like(example['input_ids'], dtype=torch.float) for example in examples]
        batch_ages = [example['ages'] for example in examples]
        batch_dates = [example['dates'] for example in examples]
        batch_visit_concept_orders = [example['visit_concept_orders'] for example in examples]
        batch_concept_values = [example['concept_values'] for example in examples]
        batch_concept_value_masks = [example['concept_value_masks'] for example in examples]
        batch_visit_segments = [example['visit_segments'] for example in examples]

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

        if has_label:
            batch_labels = [example['labels'] for example in examples]
            batch['labels'] = pad_sequence(
                batch_labels,
                batch_first=True,
                padding_value=-100
            )

        return batch
