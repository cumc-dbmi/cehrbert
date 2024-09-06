from .evaluation import main
from .evaluation_parse_args import create_evaluation_args

TRAINING_PERCENTAGE = "training_percentage"
IS_TRANSFER_LEARNING = "is_transfer_learning"
PERCENTAGES = [0.05, 0.1, 0.2, 0.4, 0.8]

if __name__ == "__main__":
    parse_args = create_evaluation_args().parse_args()
    for percentage in PERCENTAGES:
        setattr(parse_args, IS_TRANSFER_LEARNING, True)
        setattr(parse_args, TRAINING_PERCENTAGE, percentage)
        main(parse_args)
