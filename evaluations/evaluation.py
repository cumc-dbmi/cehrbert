import config.parameters as p
from evaluations.evaluation_parameters import FULL, SEQUENCE_MODEL, BASELINE_MODEL, LSTM, \
    VANILLA_BERT_LSTM, VANILLA_BERT_FEED_FORWARD, SLIDING_BERT, TEMPORAL_BERT_LSTM, \
    RANDOM_VANILLA_BERT_LSTM, HIERARCHICAL_BERT_LSTM, RANDOM_HIERARCHICAL_BERT_LSTM
from evaluations.evaluation_parse_args import create_evaluation_args
from evaluations.model_evaluators.hierarchical_bert_evaluators import *
from evaluations.model_evaluators.bert_model_evaluators import *
from evaluations.model_evaluators.sequence_model_evaluators import *
from evaluations.model_evaluators.frequency_model_evaluators import *
from utils.model_utils import *


def evaluate_sequence_models(args):
    # Load the training data
    dataset = pd.read_parquet(args.sequence_model_data_path)
    if LSTM in args.model_evaluators:
        validate_folder(args.time_attention_model_folder)
        time_attention_tokenizer_path = os.path.join(args.time_attention_model_folder,
                                                     p.tokenizer_path)
        time_aware_model_path = os.path.join(args.time_attention_model_folder,
                                             p.time_attention_model_path)
        BiLstmModelEvaluator(
            dataset=dataset,
            evaluation_folder=args.evaluation_folder,
            num_of_folds=args.num_of_folds,
            is_transfer_learning=args.is_transfer_learning,
            training_percentage=args.training_percentage,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            epochs=args.epochs,
            time_aware_model_path=time_aware_model_path,
            tokenizer_path=time_attention_tokenizer_path,
            sequence_model_name=args.sequence_model_name
        ).eval_model()

    if VANILLA_BERT_FEED_FORWARD in args.model_evaluators:
        validate_folder(args.vanilla_bert_model_folder)
        bert_tokenizer_path = os.path.join(args.vanilla_bert_model_folder,
                                           p.tokenizer_path)
        bert_model_path = os.path.join(args.vanilla_bert_model_folder,
                                       p.bert_model_validation_path)
        BertFeedForwardModelEvaluator(
            dataset=dataset,
            evaluation_folder=args.evaluation_folder,
            num_of_folds=args.num_of_folds,
            is_transfer_learning=args.is_transfer_learning,
            training_percentage=args.training_percentage,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            epochs=args.epochs,
            bert_model_path=bert_model_path,
            tokenizer_path=bert_tokenizer_path,
            is_temporal=False,
            sequence_model_name=args.sequence_model_name).eval_model()

    if SLIDING_BERT in args.model_evaluators:
        validate_folder(args.vanilla_bert_model_folder)
        bert_tokenizer_path = os.path.join(args.vanilla_bert_model_folder,
                                           p.tokenizer_path)
        bert_model_path = os.path.join(args.vanilla_bert_model_folder,
                                       p.bert_model_validation_path)
        SlidingBertModelEvaluator(
            dataset=dataset,
            evaluation_folder=args.evaluation_folder,
            num_of_folds=args.num_of_folds,
            is_transfer_learning=args.is_transfer_learning,
            training_percentage=args.training_percentage,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            epochs=args.epochs,
            bert_model_path=bert_model_path,
            tokenizer_path=bert_tokenizer_path,
            stride=args.stride,
            context_window=args.context_window,
            sequence_model_name=args.sequence_model_name).eval_model()

    if VANILLA_BERT_LSTM in args.model_evaluators:
        validate_folder(args.vanilla_bert_model_folder)
        bert_tokenizer_path = os.path.join(args.vanilla_bert_model_folder,
                                           p.tokenizer_path)
        bert_model_path = os.path.join(args.vanilla_bert_model_folder,
                                       p.bert_model_validation_path)
        BertLstmModelEvaluator(
            dataset=dataset,
            evaluation_folder=args.evaluation_folder,
            num_of_folds=args.num_of_folds,
            is_transfer_learning=args.is_transfer_learning,
            training_percentage=args.training_percentage,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            epochs=args.epochs,
            bert_model_path=bert_model_path,
            tokenizer_path=bert_tokenizer_path,
            is_temporal=False,
            sequence_model_name=args.sequence_model_name).eval_model()

    if TEMPORAL_BERT_LSTM in args.model_evaluators:
        validate_folder(args.temporal_bert_model_folder)
        temporal_bert_tokenizer_path = os.path.join(args.temporal_bert_model_folder,
                                                    p.tokenizer_path)
        temporal_bert_model_path = os.path.join(args.temporal_bert_model_folder,
                                                p.temporal_bert_validation_model_path)
        BertLstmModelEvaluator(
            dataset=dataset,
            evaluation_folder=args.evaluation_folder,
            num_of_folds=args.num_of_folds,
            is_transfer_learning=args.is_transfer_learning,
            training_percentage=args.training_percentage,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            epochs=args.epochs,
            bert_model_path=temporal_bert_model_path,
            tokenizer_path=temporal_bert_tokenizer_path,
            is_temporal=True,
            sequence_model_name=args.sequence_model_name).eval_model()

    if RANDOM_VANILLA_BERT_LSTM in args.model_evaluators:
        validate_folder(args.vanilla_bert_model_folder)
        bert_model_path = os.path.join(args.vanilla_bert_model_folder,
                                       p.bert_model_validation_path)
        bert_tokenizer_path = os.path.join(args.vanilla_bert_model_folder,
                                           p.tokenizer_path)
        visit_tokenizer_path = os.path.join(args.vanilla_bert_model_folder,
                                            p.visit_tokenizer_path)
        RandomVanillaLstmBertModelEvaluator(
            dataset=dataset,
            evaluation_folder=args.evaluation_folder,
            num_of_folds=args.num_of_folds,
            is_transfer_learning=args.is_transfer_learning,
            training_percentage=args.training_percentage,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            epochs=args.epochs,
            bert_model_path=bert_model_path,
            tokenizer_path=bert_tokenizer_path,
            visit_tokenizer_path=visit_tokenizer_path,
            is_temporal=False,
            sequence_model_name=args.sequence_model_name,
            embedding_size=args.embedding_size,
            depth=args.depth,
            num_heads=args.num_heads,
            use_time_embedding=args.use_time_embedding,
            time_embeddings_size=args.time_embeddings_size
        ).eval_model()

    if HIERARCHICAL_BERT_LSTM in args.model_evaluators:
        validate_folder(args.vanilla_bert_model_folder)
        bert_model_path = os.path.join(args.vanilla_bert_model_folder,
                                       p.bert_model_validation_path)
        bert_tokenizer_path = os.path.join(args.vanilla_bert_model_folder,
                                           p.tokenizer_path)
        HierarchicalBertEvaluator(
            dataset=dataset,
            evaluation_folder=args.evaluation_folder,
            num_of_folds=args.num_of_folds,
            is_transfer_learning=args.is_transfer_learning,
            training_percentage=args.training_percentage,
            max_num_of_visits=args.max_num_of_visits,
            max_num_of_concepts=args.max_num_of_concepts,
            batch_size=args.batch_size,
            epochs=args.epochs,
            bert_model_path=bert_model_path,
            tokenizer_path=bert_tokenizer_path,
            sequence_model_name=args.sequence_model_name
        ).eval_model()

    if RANDOM_HIERARCHICAL_BERT_LSTM in args.model_evaluators:
        validate_folder(args.vanilla_bert_model_folder)
        bert_model_path = os.path.join(args.vanilla_bert_model_folder,
                                       config.parameters.bert_model_validation_path)
        bert_tokenizer_path = os.path.join(args.vanilla_bert_model_folder,
                                           config.parameters.tokenizer_path)
        RandomHierarchicalBertEvaluator(
            dataset=dataset,
            evaluation_folder=args.evaluation_folder,
            num_of_folds=args.num_of_folds,
            is_transfer_learning=args.is_transfer_learning,
            training_percentage=args.training_percentage,
            num_of_exchanges=args.num_of_exchanges,
            max_num_of_visits=args.max_num_of_visits,
            max_num_of_concepts=args.max_num_of_concepts,
            depth=args.depth,
            num_heads=args.num_heads,
            use_time_embedding=args.use_time_embedding,
            time_embeddings_size=args.time_embeddings_size,
            embedding_size=args.embedding_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            bert_model_path=bert_model_path,
            tokenizer_path=bert_tokenizer_path,
            sequence_model_name=args.sequence_model_name
        ).eval_model()


def evaluate_baseline_models(args):
    # Load the training data
    dataset = pd.read_parquet(args.data_path)

    LogisticRegressionModelEvaluator(dataset=dataset,
                                     evaluation_folder=args.evaluation_folder,
                                     num_of_folds=args.num_of_folds,
                                     is_transfer_learning=args.is_transfer_learning,
                                     training_percentage=args.training_percentage).eval_model()
    XGBClassifierEvaluator(dataset=dataset,
                           evaluation_folder=args.evaluation_folder,
                           num_of_folds=args.num_of_folds,
                           is_transfer_learning=args.is_transfer_learning,
                           training_percentage=args.training_percentage).eval_model()


def main(args):
    if args.action == BASELINE_MODEL or args.action == FULL:
        evaluate_baseline_models(args)

    if args.action == SEQUENCE_MODEL or args.action == FULL:
        evaluate_sequence_models(args)


if __name__ == "__main__":
    main(create_evaluation_args().parse_args())
