import os
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from analyses.gpt.privacy.patient_index.base_indexer import PatientDataIndex
from analyses.gpt.privacy.member_inference import calculate_hamming_distance

validate_demographics = PatientDataIndex.validate_demographics
get_demographics = PatientDataIndex.get_demographics

RANDOM_SEE = 42


def find_nearest_neighbor(ehr, dataset):
    min_dist = 1e10
    for p in dataset.itertuples():
        dist = calculate_hamming_distance(ehr, p)
        if dist < min_dist:
            min_dist = dist
    return min_dist


def find_self_nearest_neighbor(ehr, dataset):
    min_dist = 1e10
    for p in dataset.itertuples():
        if p.person_id == ehr.person_id:
            continue
        dist = calculate_hamming_distance(ehr, p)
        if dist < min_dist:
            min_dist = dist
    return min_dist


def main(args):
    attack_data = pd.read_parquet(args.attack_data_folder)
    train_data = attack_data[attack_data.label == 1]
    evaluation_data = attack_data[attack_data.label == 0]
    synthetic_data = pd.read_parquet(args.synthetic_data_folder)

    train_data_sample = train_data.sample(args.num_of_samples, random_state=RANDOM_SEE)
    evaluation_data_sample = evaluation_data.sample(args.num_of_samples, random_state=RANDOM_SEE)
    synthetic_data_sample = synthetic_data.sample(args.num_of_samples, random_state=RANDOM_SEE)

    val1 = 0
    val2 = 0
    val3 = 0
    val4 = 0

    for p in tqdm(evaluation_data_sample.itertuples()):
        des = find_nearest_neighbor(p, synthetic_data_sample)
        dee = find_self_nearest_neighbor(p, evaluation_data_sample)
        if des > dee:
            val1 += 1

    for p in tqdm(train_data_sample.itertuples()):
        dts = find_nearest_neighbor(p, synthetic_data_sample)
        dtt = find_self_nearest_neighbor(p, train_data_sample)
        if dts > dtt:
            val3 += 1

    for p in tqdm(synthetic_data_sample.itertuples()):
        dse = find_nearest_neighbor(p, evaluation_data_sample)
        dst = find_nearest_neighbor(p, train_data_sample)
        dss = find_self_nearest_neighbor(p, synthetic_data_sample)
        if dse > dss:
            val2 += 1
        if dst > dss:
            val4 += 1

    val1 = val1 / args.num_of_samples
    val2 = val2 / args.num_of_samples
    val3 = val3 / args.num_of_samples
    val4 = val4 / args.num_of_samples

    aaes = (0.5 * val1) + (0.5 * val2)
    aaet = (0.5 * val3) + (0.5 * val4)

    nnaar = aaes - aaet

    results = {
        "NNAAE": nnaar
    }

    current_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    pd.DataFrame(results).to_parquet(
        os.path.join(args.metrics_folder, f"nearest_neighbor_inference_{current_time}.parquet")
    )


def create_argparser():
    import argparse
    parser = argparse.ArgumentParser(
        description='Nearest Neighbor Inference Analysis Arguments using the GPT model'
    )
    parser.add_argument(
        '--attack_data_folder',
        dest='attack_data_folder',
        action='store',
        help='The path for where the attack data folder',
        required=True
    )
    parser.add_argument(
        '--synthetic_data_folder',
        dest='synthetic_data_folder',
        action='store',
        help='The path for where the synthetic data folder',
        required=True
    )
    parser.add_argument(
        '--num_of_samples',
        dest='num_of_samples',
        action='store',
        type=int,
        required=False,
        default=5000
    )
    parser.add_argument(
        '--metrics_folder',
        dest='metrics_folder',
        action='store',
        help='The folder that stores the metrics',
        required=False
    )
    return parser


if __name__ == "__main__":
    main(create_argparser().parse_args())
