from data_generators.learning_objective import *


class ProbabilisticPhenotypeConditionCodeLearningObjective(
    HierarchicalMaskedLanguageModelLearningObjective
):
    def __init__(self, concept_tokenizer: ConceptTokenizer,
                 max_num_of_visits: int,
                 max_num_of_concepts: int,
                 is_training: bool):
        self._concept_tokenizer = concept_tokenizer
        self._max_num_of_visits = max_num_of_visits
        self._max_num_of_concepts = max_num_of_concepts
        self._is_training = is_training

    def get_tf_dataset_schema(self):
        output_dict_schema = {'condition_predictions': int32}
        return {}, output_dict_schema

    @validate_columns_decorator
    def process_batch(self, rows: List[RowSlicer]):
        condition_concepts = list(map(self._make_record, rows))

        # (batch_size, num_of_visits, 1)
        condition_concepts = np.squeeze(
            np.stack(pd.Series(condition_concepts) \
                     .apply(self._concept_tokenizer.encode)),
            axis=2
        )
        unused_token_id = self._concept_tokenizer.get_unused_token_id()

        # Create the condition concept masks
        # condition_concept_masks = 1 means this position has a prediction task
        condition_concept_masks = (
                condition_concepts != unused_token_id
        ).astype(int)

        output_dict = {
            'condition_predictions': np.stack(
                [condition_concepts,
                 condition_concept_masks], axis=-1
            )
        }

        return {}, output_dict

    def _make_record(self, row_slicer: RowSlicer):
        """
        A method for making a bert record for the bert data generator to yield

        :param row_slicer: a tuple containing a pandas row,
        left_index and right_index for slicing the sequences such as concepts

        :return:
        """

        row, start_index, end_index, _ = row_slicer

        concepts = self._pad_visits(row.concept_ids[start_index:end_index], '0')

        condition_masks = self._pad_visits(row.condition_masks[start_index:end_index], 0)

        random_conditions = list(
            map(
                self._pick_condition_concepts,
                zip(concepts, condition_masks)
            )
        )

        random_conditions = self._pad_visits(
            random_conditions,
            self._concept_tokenizer.unused_token
        )

        return random_conditions

    def _pick_condition_concepts(self, concepts_tuple):
        """
        Randomly pick a condition code

        :param concepts_tuple:
        :return:
        """
        concepts, condition_masks = concepts_tuple

        condition_indexes = np.argwhere(condition_masks == 1).flatten()

        if condition_indexes.size > 0:
            return concepts[np.random.choice(condition_indexes)]

        return self._concept_tokenizer.unused_token


class ProbabilisticPhenotypeLearningObjective(HierarchicalMaskedLanguageModelLearningObjective):

    @validate_columns_decorator
    def process_batch(self, rows: List[RowSlicer]):
        (
            random_concepts, concepts, dates, ages,
            visit_segments, visit_dates, visit_masks, time_interval_atts,
            visit_rank_orders
        ) = zip(*list(map(self._make_record, rows)))

        unused_token_id = self._concept_tokenizer.get_unused_token_id()

        concepts = np.stack(pd.Series(concepts) \
                            .apply(convert_to_list_of_lists) \
                            .apply(self._concept_tokenizer.encode) \
                            .apply(lambda tokens: self._pad(tokens, padded_token=unused_token_id)))

        # (batch_size, num_of_visits, 1)
        random_concepts = np.squeeze(
            np.stack(pd.Series(random_concepts) \
                     .apply(convert_to_list_of_lists) \
                     .apply(self._concept_tokenizer.encode)),
            axis=2
        )

        pat_mask = (concepts == unused_token_id).astype(int)

        time_interval_atts = np.asarray(
            self._concept_tokenizer.encode(
                np.stack(time_interval_atts).tolist()
            )
        )

        visit_masks = np.stack(visit_masks)

        visit_segments = np.stack(visit_segments)

        visit_rank_orders = np.stack(visit_rank_orders)

        # The auxiliary inputs for bert
        dates = np.stack(
            pd.Series(dates) \
                .apply(convert_to_list_of_lists) \
                .apply(lambda time_stamps: self._pad(time_stamps, padded_token=0))
        )

        ages = np.stack(
            pd.Series(ages) \
                .apply(convert_to_list_of_lists) \
                .apply(lambda time_stamps: self._pad(time_stamps, padded_token=0))
        )

        input_dict = {'pat_seq': concepts,
                      'pat_mask': pat_mask,
                      'pat_seq_time': dates,
                      'pat_seq_age': ages,
                      'visit_segment': visit_segments,
                      'visit_time_delta_att': time_interval_atts,
                      'visit_mask': visit_masks,
                      'visit_rank_order': visit_rank_orders}

        output_concept_masks = np.ones_like(random_concepts) - visit_masks

        output_dict = {
            'concept_predictions': np.stack([random_concepts, output_concept_masks], axis=-1)
        }

        return input_dict, output_dict

    def _make_record(self, row_slicer: RowSlicer):
        """
        A method for making a bert record for the bert data generator to yield

        :param row_slicer: a tuple containing a pandas row,
        left_index and right_index for slicing the sequences such as concepts

        :return:
        """

        row, start_index, end_index, _ = row_slicer

        concepts = self._pad_visits(row.concept_ids[start_index:end_index], '0')
        dates = self._pad_visits(row.dates[start_index:end_index], 0)
        ages = self._pad_visits(row.ages[start_index:end_index], 0)
        visit_segments = self._pad_visits(
            row.visit_segments[start_index:end_index], 0, False
        )
        visit_dates = self._pad_visits(
            row.visit_dates[start_index:end_index], 0, False
        )
        visit_masks = self._pad_visits(
            row.visit_masks[start_index:end_index], 1, False
        )
        # Skip the first element because there is no time interval for it
        time_interval_atts = self._pad_visits(
            row.time_interval_atts[start_index:end_index], '0',
            False)[1:]
        visit_rank_orders = self._pad_visits(
            row.visit_rank_orders[start_index:end_index],
            row.visit_rank_orders[0],
            False)

        # Lambda function for excluding the CLS token at the first position
        exclude_cls_token_fn = (
            lambda c: c[1:]
        )

        # Lambda function for randomly select a token from the given numpy array
        random_select_token_fn = (
            lambda c: c[np.random.choice(c.shape[0], 1, replace=False)]
        )

        # Randomly pick a concept from each visit for prediction (excluding the first CLS token)
        random_concepts = list(
            map(
                random_select_token_fn,
                map(exclude_cls_token_fn, row.concept_ids[start_index:end_index]),
            )
        )
        # Pad the random_concepts with num_of_visits
        random_concepts = self._pad_visits(random_concepts, '0')

        return (
            random_concepts, concepts, dates, ages, visit_segments, visit_dates, visit_masks,
            time_interval_atts, visit_rank_orders
        )
