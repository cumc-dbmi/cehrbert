import argparse
import datetime
import os
import math
import uuid
from enum import Enum
from typing import Union, List, Dict, Any
from dataclasses import dataclass, asdict
from collections import Counter
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
from transformers import GenerationConfig
from datasets import load_from_disk

from models.hf_models.tokenization_hf_cehrgpt import CehrGptTokenizer
from models.hf_models.hf_cehrgpt import CEHRGPT2LMHeadModel
from models.gpt_model import TopKStrategy, TopPStrategy, TopMixStrategy
from spark_apps.decorators.patient_event_decorator import time_month_token

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
    most_likely_time: int
    num_of_simulations: int
    time_interval_probability_table: List[Dict[str, Any]]


@dataclass
class ConceptProbability:
    concept: str
    probability: float
    num_of_simulations: int


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


def convert_to_concept_probabilities(concept_ids: List[str]) -> List[ConceptProbability]:
    # Count the occurrences of each visit concept ID
    concept_counts = Counter(concept_ids)

    # Total number of simulations
    total_simulations = len(concept_ids)

    # Create ConceptProbability objects
    concept_probabilities = []
    for concept, count in concept_counts.items():
        probability = count / total_simulations
        concept_probabilities.append(
            ConceptProbability(concept=concept, probability=probability, num_of_simulations=count))
    return concept_probabilities


def convert_att_to_time_interval(att_token: str) -> int:
    att_date_delta = None
    if att_token[0] == 'D':
        att_date_delta = int(att_token[1:])
    elif att_token[0] == 'W':
        att_date_delta = int(att_token[1:]) * 7
    elif att_token[0] == 'M':
        att_date_delta = int(att_token[1:]) * 30
    elif att_token == 'LT':
        att_date_delta = 365 * 3
    # else:
    # raise ValueError(f"Can't parse the att token {att_token}")
    # TODO: add logging here
    return att_date_delta


class TimeSensitivePredictionModel:
    def __init__(
            self,
            tokenizer: CehrGptTokenizer,
            model: CEHRGPT2LMHeadModel,
            generation_config: GenerationConfig,
            prediction_strategy: PredictionStrategy = PredictionStrategy.GREEDY_STRATEGY,
            device: torch.device = torch.device("cpu"),
            batch_size: int = 32
    ):
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.generation_config = generation_config
        self.prediction_strategy = prediction_strategy
        self.device = device
        self.batch_size = batch_size
        self.max_sequence = model.config.n_positions

    def simulate(
            self,
            partial_history: Union[np.ndarray, List[str]],
            max_new_tokens: int = 0
    ) -> List[List[str]]:

        sequence_is_demographics = len(partial_history) == 4 and partial_history[0].startswith("year")
        sequence_ends_ve = partial_history[-1] == "VE"

        if not (sequence_is_demographics | sequence_ends_ve):
            raise ValueError(
                "There are only two types of sequences allowed. 1) the sequence only contains "
                "demographics; 2) the sequence ends on VE;"
            )

        seq_length = len(partial_history)
        old_max_new_tokens = self.generation_config.max_new_tokens

        # Set this to max sequence
        self.generation_config.max_new_tokens = (
            self.model.config.n_positions - seq_length if max_new_tokens == 0 else max_new_tokens
        )
        token_ids = self.tokenizer.encode(partial_history)
        prompt = torch.tensor(token_ids).unsqueeze(0).to(self.device)

        simulated_sequences = []
        num_iters = max(
            math.ceil(self.generation_config.num_return_sequences / self.batch_size),
            1
        )
        old_num_return_sequences = self.generation_config.num_return_sequences
        self.generation_config.num_return_sequences = self.batch_size
        for _ in range(num_iters):
            with torch.no_grad():
                results = self.model.generate(
                    inputs=prompt,
                    generation_config=self.generation_config,
                )
                # Clear the cache
                torch.cuda.empty_cache()
            simulated_sequences.extend([self.tokenizer.decode(seq.cpu().numpy()) for seq in results.sequences])

        self.generation_config.num_return_sequences = old_num_return_sequences
        self.generation_config.max_new_tokens = old_max_new_tokens
        return simulated_sequences

    @staticmethod
    def extract_time_to_next_visit(
            partial_history: Union[np.ndarray, List[str]],
            simulated_seqs: List[List[str]]
    ) -> TimeToEvent:
        seq_length = len(partial_history)
        all_valid_time_intervals = []

        # Iterating through all the simulated sequences
        for simulated_seq in simulated_seqs:
            # Finding the first artificial time tokens
            for next_token in simulated_seq[seq_length:]:
                time_interval = convert_att_to_time_interval(next_token)
                if time_interval:
                    all_valid_time_intervals.append(time_interval)
                    break
        time_buckets = [time_month_token(_) for _ in all_valid_time_intervals]
        time_bucket_counter = Counter(time_buckets)
        most_common_item = time_bucket_counter.most_common(1)[0][0]
        total_count = sum(time_bucket_counter.values())
        # Generate the probability table
        probability_table = {item: count / total_count for item, count in time_bucket_counter.items()}
        sorted_probability_table = [
            {"time_interval": k, "probability": v}
            for k, v in
            sorted(probability_table.items(), key=lambda x: x[1], reverse=True)
        ]

        return TimeToEvent(
            average_time=np.mean(all_valid_time_intervals),
            standard_deviation=np.std(all_valid_time_intervals),
            most_likely_time=most_common_item,
            num_of_simulations=len(all_valid_time_intervals),
            time_interval_probability_table=sorted_probability_table
        )

    def predict_time_to_next_visit(self, partial_history: Union[np.ndarray, list]) -> TimeToEvent:
        sequences = self.simulate(
            partial_history=partial_history,
            max_new_tokens=1
        )
        return self.extract_time_to_next_visit(partial_history, sequences)

    @staticmethod
    def extract_next_visit_type(
            partial_history: Union[np.ndarray, list],
            simulated_seqs: List[List[str]]
    ) -> List[ConceptProbability]:
        next_visit_type_tokens = []
        for seq in simulated_seqs:
            for next_token in seq[len(partial_history):]:
                if next_token in VISIT_CONCEPT_IDS:
                    next_visit_type_tokens.append(next_token)
                    break
        return convert_to_concept_probabilities(next_visit_type_tokens)

    def predict_next_visit_type(
            self,
            partial_history: Union[np.ndarray, list]
    ) -> List[ConceptProbability]:
        sequences = self.simulate(
            partial_history=partial_history,
            max_new_tokens=3
        )
        return self.extract_next_visit_type(partial_history, sequences)

    @staticmethod
    def extract_events(
            partial_history: Union[np.ndarray, list],
            simulated_seqs: List[List[str]],
            only_next_visit: bool = True
    ):
        patient_history_length = len(partial_history)
        all_concepts = []
        for seq in simulated_seqs:
            for next_token in seq[patient_history_length:]:
                if only_next_visit and next_token == "VE":
                    break
                if not is_artificial_token(next_token) and next_token.isnumeric():
                    all_concepts.append(next_token)
        return convert_to_concept_probabilities(all_concepts)

    def predict_events(
            self,
            partial_history: Union[np.ndarray, list],
            only_next_visit: bool = True
    ) -> List[ConceptProbability]:

        patient_history_length = len(partial_history)
        sequences = self.simulate(
            partial_history=partial_history,
            max_new_tokens=self.model.config.n_positions - patient_history_length
        )
        return self.extract_events(partial_history, sequences, only_next_visit)

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
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    cehrgpt_tokenizer = CehrGptTokenizer.from_pretrained(args.tokenizer_folder)
    cehrgpt_model = CEHRGPT2LMHeadModel.from_pretrained(args.model_folder).eval().to(device)
    cehrgpt_model.generation_config.pad_token_id = cehrgpt_tokenizer.pad_token_id
    cehrgpt_model.generation_config.eos_token_id = cehrgpt_tokenizer.end_token_id
    cehrgpt_model.generation_config.bos_token_id = cehrgpt_tokenizer.end_token_id

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

    time_sensitive_prediction_output_folder_name = os.path.join(
        args.output_folder,
        folder_name,
        'time_sensitive_predictions'
    )

    next_visit_type_prediction_folder_name = os.path.join(
        args.output_folder,
        folder_name,
        'next_visit_type_predictions'
    )

    code_predictions_output_folder_name = os.path.join(
        args.output_folder,
        folder_name,
        'code_predictions'
    )

    os.makedirs(time_sensitive_prediction_output_folder_name, exist_ok=True)
    os.makedirs(next_visit_type_prediction_folder_name, exist_ok=True)
    os.makedirs(code_predictions_output_folder_name, exist_ok=True)

    print(f'{datetime.datetime.now()}: Loading tokenizer at {args.model_folder}')
    print(f'{datetime.datetime.now()}: Loading model at {args.model_folder}')
    print(f'{datetime.datetime.now()}: Write time sensitive predictions '
          f'to {time_sensitive_prediction_output_folder_name}')
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
        batch_size=args.batch_size,
        device=device
    )

    dataset = load_from_disk(
        args.dataset_folder
    )

    if 'test' not in dataset:
        raise ValueError(f"The dataset does not contain a test split at {args.dataset_folder}")

    test_dataset = dataset['test']

    def filter_func(examples):
        return [cehrgpt_model.config.n_positions >= _ >= 20 for _ in examples['num_of_concepts']]

    test_dataset = test_dataset.filter(
        filter_func,
        batched=True,
        batch_size=1000
    )

    person_id = 0
    tte_visit_output = []
    next_visit_prediction_output = []
    code_prediction_output = []
    for record in tqdm(test_dataset, total=len(test_dataset)):
        seq = record["concept_ids"]
        seq_length = len(seq)
        person_id += 1
        visit_counter = 0
        for index, concept_id in enumerate(seq):

            if concept_id == "VE":
                # increment the visit number by one
                visit_counter += 1

                # We only do code prediction once the cursor exceeds the half of the patient sequence.
                # In other words, we use the first half of the patient sequence to predict the second half of
                # the patient sequence.
                if seq_length - 1 > index > seq_length // 2:
                    partial_history = seq[:index + 1]
                    max_new_tokens = ts_pred_model.max_sequence - len(partial_history)
                    simulated_seqs = ts_pred_model.simulate(partial_history, max_new_tokens)

                    # Extract the predictions for time to the next visit
                    time_to_next_visit_label = convert_att_to_time_interval(seq[index + 1])
                    if time_to_next_visit_label:
                        tte = ts_pred_model.extract_time_to_next_visit(partial_history, simulated_seqs)
                        tte_visit_output.append({
                            "person_id": person_id,
                            "visit_counter": visit_counter,
                            "time_to_next_visit_label": time_to_next_visit_label,
                            "time_to_next_visit_average": tte.average_time,
                            "time_to_next_visit_std": tte.standard_deviation,
                            "time_to_next_visit_most_likely": tte.most_likely_time,
                            "time_interval_probability_table": tte.time_interval_probability_table,
                            "time_to_next_visit_simulations": tte.num_of_simulations
                        })

                    if index + 3 < seq_length and seq[index + 2] == "VS":
                        visit_type_label = seq[index + 3]
                        # Extract the predictions for the next visit type
                        visit_concept_probs = ts_pred_model.extract_next_visit_type(partial_history, simulated_seqs)
                        visit_type_prediction = sorted(
                            map(asdict, visit_concept_probs),
                            key=lambda v: v["probability"],
                            reverse=True
                        )
                        next_visit_prediction_output.append({
                            'person_id': person_id,
                            'visit_counter': visit_counter,
                            'visit_type_label': visit_type_label,
                            'next_visit_type': visit_type_label,
                            'next_visit_predictions': visit_type_prediction
                        })

                    future_event_predictions = ts_pred_model.extract_events(
                        partial_history,
                        simulated_seqs,
                        only_next_visit=False
                    )
                    future_event_predictions = sorted(
                        map(asdict, future_event_predictions), key=lambda v: v['probability'], reverse=True
                    )
                    future_event_labels = ts_pred_model.extract_events(
                        partial_history, [seq],
                        only_next_visit=False
                    )
                    future_event_labels = sorted(
                        map(asdict, future_event_labels), key=lambda v: v['probability'], reverse=True
                    )

                    code_prediction_output.append({
                        "person_id": person_id,
                        "visit_counter": visit_counter,
                        "code_labels": future_event_labels,
                        "code_predictions": future_event_predictions
                    })

                    # One patient only yield one sample
                    break

            if len(tte_visit_output) >= args.buffer_size:
                print(f'{datetime.datetime.now()}: Flushing time to visit predictions to disk')
                pd.DataFrame(
                    tte_visit_output,
                    columns=[
                        "person_id", "visit_counter", "time_to_next_visit_label", "time_to_next_visit_average",
                        "time_to_next_visit_std", "time_to_next_visit_most_likely", "time_interval_probability_table",
                        "time_to_next_visit_simulations"
                    ]
                ).to_parquet(os.path.join(time_sensitive_prediction_output_folder_name, f'{uuid.uuid4()}.parquet'))
                tte_visit_output.clear()

            if len(next_visit_prediction_output) >= args.buffer_size:
                print(f'{datetime.datetime.now()}: Flushing next visit type predictions to disk')
                pd.DataFrame(
                    next_visit_prediction_output,
                    columns=[
                        'person_id',
                        'visit_counter',
                        'visit_type_label',
                        'next_visit_type',
                        'next_visit_predictions'
                    ]
                ).to_parquet(os.path.join(next_visit_type_prediction_folder_name, f'{uuid.uuid4()}.parquet'))
                next_visit_prediction_output.clear()

            if len(code_prediction_output) >= args.buffer_size:
                print(f'{datetime.datetime.now()}: Flushing code predictions to disk')
                pd.DataFrame(
                    code_prediction_output,
                    columns=[
                        "person_id",
                        "visit_counter",
                        "code_labels",
                        "code_predictions"
                    ]
                ).to_parquet(os.path.join(code_predictions_output_folder_name, f'{uuid.uuid4()}.parquet'))
                code_prediction_output.clear()

    if len(tte_visit_output) > 0:
        print(f'{datetime.datetime.now()}: Flushing time to visit predictions to disk at Final Batch')
        pd.DataFrame(
            tte_visit_output,
            columns=[
                "person_id", "visit_counter", "time_to_next_visit_label", "time_to_next_visit_average",
                "time_to_next_visit_std", "time_to_next_visit_most_likely", "time_interval_probability_table",
                "time_to_next_visit_simulations"
            ]
        ).to_parquet(os.path.join(time_sensitive_prediction_output_folder_name, f'{uuid.uuid4()}-last.parquet'))

    if len(next_visit_prediction_output) > 0:
        print(f'{datetime.datetime.now()}: Flushing next visit type predictions to disk at Final Batch')
        pd.DataFrame(
            next_visit_prediction_output,
            columns=[
                'person_id',
                'visit_counter',
                'visit_type_label',
                'next_visit_type',
                'next_visit_predictions'
            ]
        ).to_parquet(os.path.join(next_visit_type_prediction_folder_name, f'{uuid.uuid4()}-last.parquet'))

    if len(code_prediction_output) > 0:
        print(f'{datetime.datetime.now()}: Flushing code predictions to disk at Final Batch')
        pd.DataFrame(
            code_prediction_output,
            columns=[
                "person_id",
                "visit_counter",
                "code_labels",
                "code_predictions"
            ]
        ).to_parquet(os.path.join(code_predictions_output_folder_name, f'{uuid.uuid4()}-last.parquet'))


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
        '--batch_size',
        dest='batch_size',
        action='store',
        type=int,
        required=True
    )
    parser.add_argument(
        '--buffer_size',
        dest='buffer_size',
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
