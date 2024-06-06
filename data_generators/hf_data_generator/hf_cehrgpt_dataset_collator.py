import random
from typing import Any, Dict, List
import torch
from torch.nn.utils.rnn import pad_sequence

from models.hf_models.tokenization_hf_cehrgpt import CehrGptTokenizer
from data_generators.gpt_utils import random_slice_gpt_sequence


class CehrGptDataCollator:
    def __init__(
            self, tokenizer: CehrGptTokenizer,
            max_length: int
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Pre-compute these so we can use them later on
        self.vs_token_id = tokenizer._convert_token_to_id('VS')
        self.ve_token_id = tokenizer._convert_token_to_id('VE')

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
        batch_input_ids = [
            self._convert_to_tensor(example['input_ids'])
            for example in examples
        ]
        batch_attention_mask = [
            torch.ones_like(self._convert_to_tensor(example['input_ids']), dtype=torch.float)
            for example in examples
        ]
        # Pad sequences to the max length in the batch
        batch['input_ids'] = pad_sequence(
            batch_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        ).to(torch.int)
        batch['attention_mask'] = pad_sequence(
            batch_attention_mask,
            batch_first=True,
            padding_value=0.
        )
        assert (batch['input_ids'].shape[1] <= self.max_length)
        assert (batch['attention_mask'].shape[1] <= self.max_length)
        # Prepend the START token and their associated values to the corresponding time series features
        batch['input_ids'] = torch.cat(
            [torch.full((batch_size, 1), self.tokenizer.start_token_id), batch['input_ids']],
            dim=1
        )
        # The attention_mask is set to 1 to enable attention for the CLS token
        batch['attention_mask'] = torch.cat(
            [torch.full((batch_size, 1), 1.0), batch['attention_mask']],
            dim=1
        )
        batch['labels'] = batch['input_ids'].clone()
        return batch

    def generate_start_end_index(
            self,
            record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adding the start and end indices to extract a portion of the patient sequence
        """
        seq_length = len(record['input_ids'])
        new_max_length = self.max_length - 2  # Subtract one for [START] and [END] tokens

        # Return the record directly if the actual sequence length is less than the max sequence

        if seq_length <= new_max_length:
            record['input_ids'] = torch.concat(
                [self._convert_to_tensor(record['input_ids']),
                 self._convert_to_tensor([self.tokenizer.end_token_id])]
            )
            return record

        # There is a 50% chance we randomly slice out a portion of the patient history and update the demographic
        # prompt depending on the new starting point
        if random.random() < 0.5:
            start_index, end_index, demographic_tokens = random_slice_gpt_sequence(
                record['concept_ids'],
                new_max_length
            )
            if start_index != end_index:
                record['input_ids'] = torch.concat(
                    [self._convert_to_tensor(demographic_tokens),
                     self._convert_to_tensor(record['input_ids'][start_index:end_index + 1])
                     ]
                )
                return record

        # The default employs a right truncation strategy, where the demographic prompt is reserved
        end_index = new_max_length
        for i in range(0, end_index):
            current_token = record['input_ids'][i]
            if current_token == self.ve_token_id:
                end_index = i
                break

        record['input_ids'] = record['input_ids'][0:end_index]
        return record


def concatenate(elements: List[Any]):
    assert type(elements)
