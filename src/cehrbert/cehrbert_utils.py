import re
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Union

from transformers.utils import logging

# Regular expression pattern to match inpatient attendance tokens
MEDS_CODE_PATTERN = re.compile(r".*/.*")
INPATIENT_ATT_PATTERN = re.compile(r"(?:VS-|i-)D(\d+)(?:-VE)?")
DEMOGRAPHIC_PROMPT_SIZE = 4
logger = logging.get_logger("transformers")


def construct_time_sequence(
    concept_ids: List[str], epoch_times: Optional[List[Union[int, float]]] = None
) -> List[float]:
    if epoch_times is not None:
        return epoch_times

    if concept_ids[0].lower().startswith("year"):
        year_str = concept_ids[0].split(":")[1]
    else:
        year_str = "1985"

    datetime_cursor = datetime(int(year_str), month=1, day=1, hour=0, minute=0, second=0).replace(tzinfo=timezone.utc)
    epoch_times = []
    for concept_id in concept_ids:
        if is_att_token(concept_id):
            att_days = extract_time_interval_in_days(concept_id)
            datetime_cursor += timedelta(days=att_days)
        epoch_times.append(datetime_cursor.timestamp())
    return epoch_times


def is_att_token(token: str):
    """
    Check if the token is an attention token.

    :param token: Token to check.
    :return: True if the token is an attention token, False otherwise.
    """
    if bool(re.match(r"^D\d+", token)):  # day tokens
        return True
    elif bool(re.match(r"^W\d+", token)):  # week tokens
        return True
    elif bool(re.match(r"^M\d+", token)):  # month tokens
        return True
    elif bool(re.match(r"^Y\d+", token)):  # year tokens
        return True
    elif token == "LT":
        return True
    elif token[:3] == "VS-":  # VS-D7-VE
        return True
    elif token[:2] == "i-" and not token.startswith("i-H"):  # i-D7 and exclude hour tokens
        return True
    return False


def extract_time_interval_in_days(token: str):
    """
    Extract the time interval in days from a token.

    :param token: Token to extract from.
    :return: Time interval in days.
    :raises ValueError: If the token is invalid.
    """
    try:
        if token[0] == "D":  # day tokens
            return int(token[1:])
        elif token[0] == "W":  # week tokens
            return int(token[1:]) * 7
        elif token[0] == "M":  # month tokens
            return int(token[1:]) * 30
        elif token[0] == "Y":  # year tokens
            return int(token[1:]) * 365
        elif token == "LT":
            return 365 * 3
        elif token[:3] == "VS-":  # VS-D7-VE
            part = token.split("-")[1]
            if part.startswith("LT"):
                return 365 * 3
            return int(part[1:])
        elif token[:2] == "i-":  # i-D7
            part = token.split("-")[1]
            if part.startswith("LT"):
                return 365 * 3
            return int(token.split("-")[1][1:])
    except Exception:
        raise ValueError(f"Invalid time token: {token}")
    raise ValueError(f"Invalid time token: {token}")
