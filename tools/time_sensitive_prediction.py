import argparse
import datetime
import os
import uuid
import json
from enum import Enum
from typing import Union, List
from dataclasses import dataclass
from collections import Counter
from tqdm import tqdm

import numpy as np
import torch
from models.hf_models.tokenization_hf_cehrgpt import CehrGptTokenizer
from transformers import GenerationConfig
from models.hf_models.hf_cehrgpt import CEHRGPT2LMHeadModel

from models.gpt_model import TopKStrategy, TopPStrategy, TopMixStrategy

VISIT_CONCEPT_IDS = [
    '9202', '9203', '581477', '9201', '5083', '262', '38004250', '0', '8883', '38004238', '38004251',
    '38004222', '38004268', '38004228', '32693', '8971', '38004269', '38004193', '32036', '8782'
]

# TODO: fill this in
DISCHARGE_CONCEPT_IDS = [

]


class PredictionStrategy(Enum):
    GREEDY_STRATEGY = "greedy_strategy"


@dataclass
class TimeToEvent:
    average_time: float
    standard_deviation: float
    num_of_simulations: int


@dataclass
class ConceptProbability:
    concept: str
    probability: float
    num_of_simulations: int


class TimeSensitivePredictionModel:
    def __init__(
            self,
            tokenizer: CehrGptTokenizer,
            model: CEHRGPT2LMHeadModel,
            generation_config: GenerationConfig,
            prediction_strategy: PredictionStrategy = PredictionStrategy.GREEDY_STRATEGY,
            device: torch.device = torch.device("cpu")
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.generation_config = generation_config
        self.prediction_strategy = prediction_strategy
        self.device = device

    def predict_time_to_next_visit(self, partial_history: Union[np.ndarray, list]) -> TimeToEvent:
        """
            Predict the time interval to the next visit based on a partial patient history.

            This method generates predictions for the time interval until the next visit event (VE)
            using a sequence model. The prediction is based on a provided partial history of
            patient data.

            Parameters:
            -----------
            partial_history : Union[np.ndarray, list]
                The partial history of the patient represented as a sequence of tokens. It can be
                a PyTorch IntTensor, a NumPy array, or a list.

            Returns:
            --------
            TimeToEvent
                An object containing the average time to the next visit, the standard deviation
                of the predicted times, and the number of simulations used to derive the predictions.

            Notes:
            ------
            - The method temporarily modifies the `max_new_tokens` setting of the generation configuration.
            - The special token "LT" is interpreted as a time interval of 1080 units (days).

            Example Usage:
            --------------
            >>> model = MyModel()
            >>> partial_history = [1, 2, 3, 4, 5]
            >>> time_to_event = model.predict_time_to_next_visit(partial_history)
            >>> print(time_to_event.average_time)
            >>> print(time_to_event.standard_deviation)
            >>> print(time_to_event.num_of_simulations)
            """

        if partial_history[-1] != "VE":
            raise ValueError("The last token of the patient history must be VE.")

        seq_length = len(partial_history)
        old_max_new_tokens = self.generation_config.max_new_tokens
        # Set this to 1 because we only want to predict the time interval after the VE token
        self.generation_config.max_new_tokens = 1
        token_ids = self.tokenizer.encode(partial_history)
        prompt = torch.tensor(token_ids).unsqueeze(0).to(self.device)
        results = self.model.generate(
            inputs=prompt,
            generation_config=self.generation_config,
        )
        self.generation_config.max_new_tokens = old_max_new_tokens

        all_valid_time_intervals = []
        for seq in results.sequences:
            concept_ids = self.tokenizer.decode(seq.cpu().numpy())
            if seq_length < len(concept_ids):
                next_token = concept_ids[seq_length]
                time_interval = None
                if next_token.startswith("D"):
                    time_interval = int(next_token[1:])
                elif next_token == "LT":
                    time_interval = 1080
                if time_interval:
                    all_valid_time_intervals.append(time_interval)

        return TimeToEvent(
            average_time=np.mean(all_valid_time_intervals),
            standard_deviation=np.std(all_valid_time_intervals),
            num_of_simulations=len(all_valid_time_intervals)
        )

    def predict_next_visit_type(
            self,
            partial_history: Union[np.ndarray, list]
    ) -> List[ConceptProbability]:

        sequence_is_demographics = len(partial_history) == 4 and partial_history[0].startswith("year")
        sequence_ends_with_time_interval = partial_history[-1].startswith("D")
        sequence_ends_ve = partial_history[-1] == "VE"

        if not (sequence_is_demographics | sequence_ends_with_time_interval | sequence_ends_ve):
            raise ValueError(
                "There are only three types of sequences allowed. 1) the sequence only contains "
                "demographics; 2) the sequence ends on VE; 3) the sequence ends on a time interval"
            )

        seq_length = len(partial_history)
        old_max_new_tokens = self.generation_config.max_new_tokens
        # Set this to 3 because we only want to generate three more tokens, which must cover the next visit type token
        self.generation_config.max_new_tokens = 3
        token_ids = self.tokenizer.encode(partial_history)
        prompt = torch.tensor(token_ids).unsqueeze(0).to(self.device)
        results = self.model.generate(
            inputs=prompt,
            generation_config=self.generation_config,
        )
        self.generation_config.max_new_tokens = old_max_new_tokens
        next_visit_type_tokens = []
        for seq in results.sequences:
            concept_ids = self.tokenizer.decode(seq.cpu().numpy())
            for next_token in concept_ids[seq_length:]:
                if next_token in VISIT_CONCEPT_IDS:
                    next_visit_type_tokens.append(next_token)
                    break
        return self.convert_to_concept_probabilities(next_visit_type_tokens)

    def predict_events(
            self,
            partial_history: Union[np.ndarray, list],
            only_next_visit: bool = True
    ) -> List[ConceptProbability]:

        sequence_is_demographics = len(partial_history) == 4 and partial_history[0].startswith("year")
        sequence_ends_with_time_interval = partial_history[-1].startswith("D")
        sequence_ends_ve = partial_history[-1] == "VE"

        if not (sequence_is_demographics | sequence_ends_with_time_interval | sequence_ends_ve):
            raise ValueError(
                "There are only three types of sequences allowed. 1) the sequence only contains "
                "demographics; 2) the sequence ends on VE; 3) the sequence ends on a time interval"
            )

        seq_length = len(partial_history)
        old_max_new_tokens = self.generation_config.max_new_tokens
        # Set this to 3 because we only want to generate three more tokens, which must cover the next visit type token
        self.generation_config.max_new_tokens = self.model.config.n_positions - seq_length
        token_ids = self.tokenizer.encode(partial_history)
        prompt = torch.tensor(token_ids).unsqueeze(0).to(self.device)
        results = self.model.generate(
            inputs=prompt,
            generation_config=self.generation_config,
        )
        self.generation_config.max_new_tokens = old_max_new_tokens

        all_concepts = []
        for seq in results.sequences:
            generated_seq = self.tokenizer.decode(seq.cpu().numpy())
            for next_token in generated_seq[seq_length:]:
                if only_next_visit and next_token == "VE":
                    break
                if not self.is_artificial_token(next_token):
                    all_concepts.append(next_token)
        return self.convert_to_concept_probabilities(all_concepts)

    @staticmethod
    def is_artificial_token(token) -> bool:
        if token in VISIT_CONCEPT_IDS:
            return True
        if token in DISCHARGE_CONCEPT_IDS:
            return True
        if token in ["VS", "VE"]:
            return True
        if token.startswith("D"):
            return True
        if token == "LT":
            return True
        return False

    @staticmethod
    def convert_to_concept_probabilities(visit_concept_ids: List[str]) -> List[ConceptProbability]:
        # Count the occurrences of each visit concept ID
        concept_counts = Counter(visit_concept_ids)

        # Total number of simulations
        total_simulations = len(visit_concept_ids)

        # Create ConceptProbability objects
        concept_probabilities = []
        for concept, count in concept_counts.items():
            probability = count / total_simulations
            concept_probabilities.append(
                ConceptProbability(concept=concept, probability=probability, num_of_simulations=count))

        return concept_probabilities

    @staticmethod
    def get_generation_config(
            tokenizer: CehrGptTokenizer,
            max_length: int,
            num_return_sequences: int,
            top_p: float = 1.0,
            top_k: int = 300,
            temperature: float = 1.0
    ) -> GenerationConfig:
        return GenerationConfig(
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            bos_token_id=tokenizer.end_token_id,
            eos_token_id=tokenizer.end_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=False,
            output_hidden_states=False,
            output_scores=False,
            renormalize_logits=True
        )


def main(
        args
):
    from datasets import load_from_disk

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(args.tokenizer_folder)
    cehrgpt_model = CEHRGPT2LMHeadModel.from_pretrained(args.model_folder).eval().to(device)

    if args.sampling_strategy == TopKStrategy.__name__:
        folder_name = f'top_k{args.top_k}'
        args.top_p = 1.0
    elif args.sampling_strategy == TopPStrategy.__name__:
        folder_name = f'top_p{int(args.top_p * 1000)}'
        args.top_k = cehrgpt_tokenizer.vocab_size
    elif args.sampling_strategy == TopMixStrategy.__name__:
        folder_name = f'top_mix_p{int(args.top_p * 1000)}_k{args.top_k}'
    else:
        raise RuntimeError(
            'sampling_strategy has to be one of the following two options TopKStrategy or TopPStrategy'
        )

    output_folder_name = os.path.join(
        args.output_folder,
        folder_name,
        'time_sensitive_predictions.json'
    )

    print(f'{datetime.datetime.now()}: Loading tokenizer at {args.model_folder}')
    print(f'{datetime.datetime.now()}: Loading model at {args.model_folder}')
    print(f'{datetime.datetime.now()}: Write time sensitive predictions to {output_folder_name}')
    print(f'{datetime.datetime.now()}: Top P {args.top_p}')
    print(f'{datetime.datetime.now()}: Top K {args.top_k}')
    print(f'{datetime.datetime.now()}: Loading dataset_folder at {args.dataset_folder}')

    generation_config = TimeSensitivePredictionModel.get_generation_config(
        tokenizer=cehrgpt_tokenizer,
        max_length=cehrgpt_model.config.n_positions,
        num_return_sequences=args.num_return_sequences,
        top_p=args.top_p,
        top_k=args.top_k
    )
    ts_pred_model = TimeSensitivePredictionModel(
        tokenizer=cehrgpt_tokenizer,
        model=cehrgpt_model,
        generation_config=generation_config,
        device=device
    )

    dataset = load_from_disk(
        args.dataset_folder
    )

    if 'test' not in dataset:
        raise ValueError(f"The dataset does not contain a test split at {args.dataset_folder}")

    test_dataset = dataset['test']

    def filter_func(examples):
        return [_ <= cehrgpt_model.config.n_positions for _ in examples['num_of_concepts']]

    test_dataset = test_dataset.filter(
        filter_func,
        batched=True,
        batch_size=1000
    ).select(range(20))

    person_id = 0
    output = dict()
    for record in tqdm(test_dataset, total=len(test_dataset)):
        person_id += 1
        seq = record["concept_ids"]
        visit_counter = 0
        att_predictions = dict()
        for index, concept_id in enumerate(seq):
            if concept_id == "VE" and index < len(seq) - 1:
                next_token = seq[index + 1]
                time_to_next_visit_label = None
                if next_token.startswith("D"):
                    time_to_next_visit_label = int(next_token[1:])
                elif next_token == "LT":
                    time_to_next_visit_label = 1080
                if time_to_next_visit_label:
                    with torch.no_grad():
                        tte = ts_pred_model.predict_time_to_next_visit(seq[:index + 1])
                    # Clear the cache
                    torch.cuda.empty_cache()
                    att_predictions[visit_counter] = {
                        "time_to_next_visit_label": time_to_next_visit_label,
                        "time_to_next_visit_average": tte.average_time,
                        "time_to_next_visit_std": tte.standard_deviation,
                        "time_to_next_visit_simulations": tte.num_of_simulations
                    }
                visit_counter += 1
        output[person_id] = att_predictions

    with open(output_folder_name, 'w') as json_file:
        json.dump(output, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for time sensitive predictions')
    parser.add_argument(
        '--dataset_folder',
        dest='dataset_folder',
        action='store',
        help='The path for your dataset',
        required=True
    )
    parser.add_argument(
        '--tokenizer_folder',
        dest='tokenizer_folder',
        action='store',
        help='The path for your model_folder',
        required=True
    )
    parser.add_argument(
        '--model_folder',
        dest='model_folder',
        action='store',
        help='The path for your model_folder',
        required=True
    )
    parser.add_argument(
        '--output_folder',
        dest='output_folder',
        action='store',
        help='The path for your generated data',
        required=True
    )
    parser.add_argument(
        '--sampling_strategy',
        dest='sampling_strategy',
        action='store',
        choices=[TopPStrategy.__name__, TopKStrategy.__name__, TopMixStrategy.__name__],
        help='Pick the sampling strategy between top_k and top_p',
        required=True
    )
    parser.add_argument(
        '--num_return_sequences',
        dest='num_return_sequences',
        action='store',
        type=int,
        required=True
    )
    parser.add_argument(
        '--top_k',
        dest='top_k',
        action='store',
        default=100,
        type=int,
        help='The number of top concepts to sample',
        required=False
    )
    parser.add_argument(
        '--top_p',
        dest='top_p',
        action='store',
        default=1.0,
        type=float,
        help='The accumulative probability of top concepts to sample',
        required=False
    )
    main(parser.parse_args())
