import argparse


def create_evaluation_args():
    main_parser = argparse.ArgumentParser(description="Arguments for evaluating the models")
    main_parser.add_argument(
        "-d",
        "--data_path",
        dest="data_path",
        action="store",
        help="The training data path",
        required=True,
    )
    main_parser.add_argument(
        "--patient_splits_folder",
        dest="patient_splits_folder",
        action="store",
        help="The test person_ids data",
        required=False,
    )
    main_parser.add_argument(
        "-ef",
        "--evaluation_folder",
        dest="evaluation_folder",
        action="store",
        required=True,
    )
    main_parser.add_argument(
        "-n",
        "--num_of_folds",
        dest="num_of_folds",
        action="store",
        required=False,
        type=int,
        default=4,
    )
    main_parser.add_argument("--is_transfer_learning", dest="is_transfer_learning", action="store_true")
    main_parser.add_argument(
        "--training_percentage",
        dest="training_percentage",
        required=False,
        action="store",
        type=float,
        default=1.0,
    )
    return main_parser
