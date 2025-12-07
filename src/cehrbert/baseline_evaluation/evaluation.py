import pandas as pd

from .evaluation_parse_args import create_evaluation_args
from .model_evaluators.frequency_model_evaluators import (
    LogisticRegressionModelEvaluator,
    XGBClassifierEvaluator,
)


def evaluate_baseline_models(args):
    # Load the training data
    dataset = pd.read_parquet(args.data_path)
    test_person_ids = None
    if args.patient_splits_folder:
        patient_splits = pd.read_parquet(args.patient_splits_folder)
        test_person_ids = patient_splits[patient_splits.split == "test"]

    LogisticRegressionModelEvaluator(
        dataset=dataset,
        evaluation_folder=args.evaluation_folder,
        num_of_folds=args.num_of_folds,
        is_transfer_learning=args.is_transfer_learning,
        training_percentage=args.training_percentage,
        k_fold_test=args.k_fold_test,
        test_person_ids=test_person_ids,
    ).eval_model()

    XGBClassifierEvaluator(
        dataset=dataset,
        evaluation_folder=args.evaluation_folder,
        num_of_folds=args.num_of_folds,
        is_transfer_learning=args.is_transfer_learning,
        training_percentage=args.training_percentage,
        k_fold_test=args.k_fold_test,
        test_person_ids=test_person_ids,
    ).eval_model()


def main(args):
    evaluate_baseline_models(args)


if __name__ == "__main__":
    main(create_evaluation_args().parse_args())
