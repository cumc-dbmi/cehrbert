import json
import os
import re

LEGACY_MODEL_CHECKPOINT_PATTERN = re.compile(r"bert_model_(\d{2})_(\d+\.\d+)\.h5$")
EPOCH_CHECKPOINT_PATTERN = re.compile(r"_epoch_(\d{2})_batch_final\.h5$")
BATCH_CHECKPOINT_PATTERN = re.compile(r".*_epoch_(\d{2})_batch_(\d+)\.h5$")


def checkpoint_exists(model_folder, checkpoint_name):
    if checkpoint_name:
        existing_model_path = os.path.join(model_folder, checkpoint_name)
        return os.path.exists(existing_model_path)
    return False


def get_checkpoint_epoch(checkpoint_path):
    epoch = get_latest_checkpoint_legacy_model_epoch(checkpoint_path)
    if not epoch:
        epoch = get_latest_batch_checkpoint_epoch(checkpoint_path)
    if not epoch:
        epoch = get_latest_epoch_checkpoint_epoch(checkpoint_path)

    if epoch:
        return epoch

    raise RuntimeError(
        f"The model checkpoint at {checkpoint_path} does not match any patterns below:\n"
        f"{LEGACY_MODEL_CHECKPOINT_PATTERN.pattern}\n"
        f"{EPOCH_CHECKPOINT_PATTERN.pattern}\n"
        f"{BATCH_CHECKPOINT_PATTERN.pattern}\n"
    )


def find_latest_checkpoint_path(checkpoint_dir):
    # Try to find the checkpoint with the legacy model naming convention
    legacy_checkpoint_path_dict = find_latest_checkpoint_legacy_model_path(checkpoint_dir)
    if legacy_checkpoint_path_dict:
        return legacy_checkpoint_path_dict["checkpoint_path"]

    # Try to find the checkpoints associated with batch or epoch
    epoch_checkpoint_path_dict = find_latest_epoch_checkpoint_path(checkpoint_dir)
    batch_checkpoint_path_dict = find_latest_batch_checkpoint_path(checkpoint_dir)

    # We always prefer the epoch checkpoint over the batch checkpoint if they have the same epoch
    if epoch_checkpoint_path_dict and batch_checkpoint_path_dict:
        if epoch_checkpoint_path_dict["epoch"] >= batch_checkpoint_path_dict["epoch"]:
            return epoch_checkpoint_path_dict["checkpoint_path"]
        else:
            return batch_checkpoint_path_dict["checkpoint_path"]

    if epoch_checkpoint_path_dict:
        return epoch_checkpoint_path_dict["checkpoint_path"]

    if batch_checkpoint_path_dict:
        return batch_checkpoint_path_dict["checkpoint_path"]

    raise RuntimeError(
        f"Could not discover any model checkpoint in {checkpoint_dir} matching patterns\n"
        f"{LEGACY_MODEL_CHECKPOINT_PATTERN.pattern}\n"
        f"{EPOCH_CHECKPOINT_PATTERN.pattern}\n"
        f"{BATCH_CHECKPOINT_PATTERN.pattern}\n"
    )


def get_latest_checkpoint_legacy_model_epoch(filename):
    match = LEGACY_MODEL_CHECKPOINT_PATTERN.search(filename)
    if match:
        epoch, _ = match.groups()
        epoch = int(epoch)
        return epoch
    return None


def find_latest_checkpoint_legacy_model_path(checkpoint_dir):
    # List all files in the checkpoint directory
    files = os.listdir(checkpoint_dir)

    # Filter files that match the checkpoint pattern and extract epoch, loss values
    checkpoints = []
    for filename in files:
        match = LEGACY_MODEL_CHECKPOINT_PATTERN.search(filename)
        if match:
            epoch, loss = match.groups()
            epoch = int(epoch)
            loss = float(loss)
            checkpoints.append((epoch, loss, filename))

    # Sort the checkpoints by epoch in descending order (latest first), then by loss in ascending order (lowest first)
    checkpoints.sort(reverse=True, key=lambda x: (x[0], -x[1]))
    if checkpoints:
        return {
            "epoch": checkpoints[0][0],
            "loss": checkpoints[0][1],
            "checkpoint_path": checkpoints[0][2],
        }
    return None


def get_latest_epoch_checkpoint_epoch(filename):
    match = EPOCH_CHECKPOINT_PATTERN.search(filename)
    if match:
        epoch = int(match.group(1))
        return epoch
    return None


def find_latest_epoch_checkpoint_path(checkpoint_dir):
    # List all files in the checkpoint directory
    files = os.listdir(checkpoint_dir)

    # Filter files that match the checkpoint pattern and extract epoch numbers
    checkpoints = []
    for filename in files:
        match = EPOCH_CHECKPOINT_PATTERN.search(filename)
        if match:
            epoch = int(match.group(1))  # Convert the matched epoch number to an integer
            checkpoints.append((epoch, filename))

    # Sort the checkpoints by epoch in descending order (to get the latest one first)
    checkpoints.sort(reverse=True, key=lambda x: x[0])
    if checkpoints:
        return {"epoch": checkpoints[0][0], "checkpoint_path": checkpoints[0][1]}
    return None


def get_latest_batch_checkpoint_epoch(filename):
    step_match = BATCH_CHECKPOINT_PATTERN.search(filename)
    if step_match:
        epoch, _ = map(int, step_match.groups())
        return epoch
    return None


def find_latest_batch_checkpoint_path(checkpoint_dir):
    # List all files in the checkpoint directory
    files = os.listdir(checkpoint_dir)

    # Filter files that match the checkpoint pattern and extract epoch, batch numbers
    checkpoints = []
    for filename in files:
        step_match = BATCH_CHECKPOINT_PATTERN.search(filename)
        if step_match:
            epoch, batch = map(int, step_match.groups())
            checkpoints.append((epoch, batch, filename))

    # Sort the checkpoints by epoch and batch in descending order
    checkpoints.sort(reverse=True, key=lambda x: (x[0], x[1]))
    # Return the latest checkpoint filename, or None if no checkpoint was found
    if checkpoints:
        return {
            "epoch": checkpoints[0][0],
            "batch": checkpoints[0][1],
            "checkpoint_path": checkpoints[0][2],
        }
    return None


def find_tokenizer_path(model_folder: str):
    import glob

    file_path = os.path.join(model_folder, MODEL_CONFIG_FILE)
    if os.path.exists(file_path):
        # Open the JSON file for reading
        with open(file_path, "r") as file:
            model_config = json.load(file)
            tokenizer_name = model_config["tokenizer"]
            tokenizer_path = os.path.join(model_folder, tokenizer_name)
            return tokenizer_path
    else:
        for candidate_name in glob.glob(os.path.join(model_folder, "*tokenizer.pickle")):
            if "visit_tokenizer.pickle" not in candidate_name:
                return os.path.join(model_folder, candidate_name)

    raise RuntimeError(f"Could not discover any tokenizer in {model_folder} matching the pattern *tokenizer.pickle")


def find_visit_tokenizer_path(model_folder: str):
    import glob

    file_path = os.path.join(model_folder, MODEL_CONFIG_FILE)
    if os.path.exists(file_path):
        # Open the JSON file for reading
        with open(file_path, "r") as file:
            model_config = json.load(file)
            visit_tokenizer_name = model_config["visit_tokenizer"]
            visit_tokenizer_path = os.path.join(model_folder, visit_tokenizer_name)
            return visit_tokenizer_path
    else:
        for candidate_name in glob.glob(os.path.join(model_folder, "*visit_tokenizer.pickle")):
            return os.path.join(model_folder, candidate_name)

    raise RuntimeError(
        f"Could not discover any tokenizer in {model_folder} matching the pattern *_visit_tokenizer.pickle"
    )


MODEL_CONFIG_FILE = "model_config.json"
