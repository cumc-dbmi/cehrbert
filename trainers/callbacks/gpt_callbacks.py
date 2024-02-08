import copy
import datetime
import json
import os
import random

import numpy as np
import pandas as pd

import tensorflow as tf

from data_generators.tokenizer import ConceptTokenizer
from models.gpt_model import GptInferenceModel, TopKStrategy


class StepValidationCallBack(tf.keras.callbacks.Callback):
    def __init__(
            self,
            val_data_generator,
            save_freq,
            model_folder
    ):
        self.val_data_generator = val_data_generator
        self.save_freq = save_freq
        self.model_folder = model_folder
        self.metrics_folder = os.path.join(self.model_folder, 'metrics')
        self.epoch = 0

        if not os.path.exists(self.metrics_folder):
            os.makedirs(self.metrics_folder)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_batch_end(self, batch, logs=None):
        if self.val_data_generator is None or batch == 0 or batch % self.save_freq != 0:
            return
        val_steps_per_epoch = self.val_data_generator.get_steps_per_epoch()
        val_dataset = tf.data.Dataset.from_generator(
            self.val_data_generator.create_batch_generator,
            output_types=(self.val_data_generator.get_tf_dataset_schema())
        ).prefetch(tf.data.experimental.AUTOTUNE)
        results = self.model.evaluate(val_dataset, steps=val_steps_per_epoch)
        metrics = copy.deepcopy(logs)
        val_loss, val_perplexity = results
        metrics['val_loss'] = val_loss
        metrics['val_perplexity'] = val_perplexity
        # Convert and write JSON object to file
        metric_file = os.path.join(self.metrics_folder, f"epoch-{self.epoch}-batch-{batch}-metrics.json")
        with open(metric_file, "w") as outfile:
            json.dump(metrics, outfile)


class ComputeMarginalDistribution(tf.keras.callbacks.Callback):
    def __init__(
            self,
            demographic_info,
            max_seq,
            concept_tokenizer: ConceptTokenizer,
            concept_map: dict,
            batch_size,
            num_of_patients=1024,
            top_k=10,
            print_every=1
    ):
        self.demographic_info = demographic_info
        self.max_seq = max_seq
        self.concept_tokenizer = concept_tokenizer
        self.concept_map = concept_map
        self.print_every = print_every
        self.k = top_k
        self.batch_size = batch_size
        self.num_of_patients = num_of_patients

    def detokenize(self, number):
        concept_id = self.concept_tokenizer.decode([[number]])[0]
        if concept_id in self.concept_map:
            return self.concept_map[concept_id]
        return concept_id

    def on_batch_end(self, batch, logs=None):
        if batch == 0 or batch % self.print_every != 0:
            return
        inference_model = GptInferenceModel(
            self.model,
            tokenizer=self.concept_tokenizer,
            context_window=self.max_seq,
            sampling_strategy=TopKStrategy(top_k=self.k)
        )

        num_of_batches = self.num_of_patients // self.batch_size + 1
        sequence_to_flush = []
        for i in range(num_of_batches):

            print(f'{datetime.datetime.now()}: Patient generation batch {i} started')

            start_tokens = np.tile(
                np.asarray([[self.concept_tokenizer.get_start_token_id()]]),
                [self.batch_size, 1]
            )
            random_prompts = random.sample(
                self.demographic_info,
                self.batch_size
            )
            prompt_batch = np.hstack([start_tokens, random_prompts])
            _, length = np.shape(
                prompt_batch
            )

            prompt_batch = tf.cast(prompt_batch, dtype=tf.int32)

            prompt_batch = inference_model(
                prompt_batch
            )
            for seq in prompt_batch.tolist():
                seq_copy = []
                for token in seq:
                    if token == self.concept_tokenizer.get_end_token_id():
                        break
                    seq_copy.append(self.detokenize(token))
                sequence_to_flush.append({'token_ids': seq_copy})

        generated_patient_sequences = pd.DataFrame(
            sequence_to_flush,
            columns=['token_ids']
        )
        dist = generated_patient_sequences.token_ids.explode().value_counts() / len(generated_patient_sequences)
        print(f'{datetime.datetime.now()}: The marginal distribution is below:\n {dist.head(60)}\n')
        txt = '\n'.join(sequence_to_flush[0]['token_ids'])
        print(f'{datetime.datetime.now()}: The generated patient sequence:\n{txt}\n')


class PatientHistoryGenerator(tf.keras.callbacks.Callback):
    def __init__(
            self,
            demographic_info,
            max_seq,
            concept_tokenizer: ConceptTokenizer,
            concept_map: dict,
            top_k=10,
            temperature=1.0,
            print_every=1
    ):
        self.max_seq = max_seq
        self.concept_tokenizer = concept_tokenizer
        self.concept_map = concept_map
        self.print_every = print_every
        self.k = top_k
        self.temperature = temperature
        self.genders = np.squeeze(concept_tokenizer.encode([
            '8532',  # FEMALE,
            '8507'  # MALE
        ])).tolist()

        self.races = np.squeeze(concept_tokenizer.encode([
            '0',  # No matching concept
            '8527',  # white
            '8552',  # Unknown
            '8516',  # Black or African American,
            '44814653',  # Unknown
            '8522',  # Other Race
            '8515',  # Asian
        ])).tolist()

        self.starting_ages = np.squeeze(concept_tokenizer.encode(
            list(map(str, range(30, 50))) + list(map(lambda a: f'age:{a}', range(30, 50)))
        )).tolist()

        self.starting_years = np.squeeze(concept_tokenizer.encode(
            list(map(str, range(2000, 2020))) + list(map(lambda a: f'year:{a}', range(2000, 2020)))
        )).tolist()

    def detokenize(self, number):
        concept_id = self.concept_tokenizer.decode([[number]])[0]
        if concept_id in self.concept_map:
            return self.concept_map[concept_id]
        return concept_id

    def on_batch_end(self, batch, logs=None):
        if batch == 0 or batch % self.print_every != 0:
            return
        print(f'\nGenerating text for {batch}\n')

        inference_model = GptInferenceModel(
            self.model,
            tokenizer=self.concept_tokenizer,
            context_window=self.max_seq,
            sampling_strategy=TopKStrategy(top_k=self.k, temperature=self.temperature)
        )
        start_tokens = [
            self.concept_tokenizer.get_start_token_id(),
            random.sample(self.starting_years, 1)[0],
            random.sample(self.starting_ages, 1)[0],
            random.sample(self.genders, 1)[0],
            random.sample(self.races, 1)[0]
        ]
        start_tokens = tf.reshape(
            start_tokens,
            (1, -1)
        )
        prompt_batch = inference_model(
            start_tokens
        )

        txt = '\n'.join(
            [self.detokenize(_) for _ in prompt_batch[0]]
        )

        print(f"generated text:\n{txt}\n")
