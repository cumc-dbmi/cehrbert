from torch.nn.utils.rnn import pad_sequence
import torch
from models.hf_models.tokenization_hf_cehrbert import CehrBertTokenizer


class HFCehrBertDataCollator:
    def __init__(self, tokenizer: CehrBertTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        batch = {}
        # Assume that each example in the batch is a dictionary with 'input_ids' and 'attention_mask'
        batch_input_ids = [torch.tensor(example['input_ids'], dtype=torch.long) for example in examples]
        batch_attention_mask = [torch.tensor(example['attention_mask'], dtype=torch.long) for example in examples]

        # Pad sequences to the max length in the batch
        batch['input_ids'] = pad_sequence(batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        batch['attention_mask'] = pad_sequence(batch_attention_mask, batch_first=True,
                                               padding_value=0)  # 0 where we want attention to ignore

        # Additional fields can be added here, such as 'token_type_ids' if required by your model architecture
        if 'token_type_ids' in examples[0]:
            batch_token_type_ids = [torch.tensor(example['token_type_ids'], dtype=torch.long) for example in examples]
            batch['token_type_ids'] = pad_sequence(batch_token_type_ids, batch_first=True,
                                                   padding_value=self.tokenizer.pad_token_type_id)

        return batch
