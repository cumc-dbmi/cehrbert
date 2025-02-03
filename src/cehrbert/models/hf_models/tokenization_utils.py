import collections
import json
import pickle
from functools import partial
from typing import Any, Dict

from cehrbert_data.const.common import NA

from cehrbert.utils.stat_utils import TruncatedOnlineStatistics


def load_json_file(json_file):
    try:
        with open(json_file, "r", encoding="utf-8") as reader:
            file_contents = reader.read()
            parsed_json = json.loads(file_contents)
            return parsed_json
    except Exception as e:
        raise RuntimeError(f"Can't load the json file at {json_file} due to {e}")


def agg_helper(*args, map_func):
    result = map_func(*args)
    return {"data": [pickle.dumps(result)]}


def map_statistics(batch: Dict[str, Any], capacity=100, value_outlier_std=2.0) -> Dict[str, Any]:
    if "units" in batch:
        concept_value_units = batch["units"]
    else:
        concept_value_units = [[NA for _ in cons] for cons in batch["concept_ids"]]
    numeric_stats_by_lab = collections.defaultdict(
        partial(TruncatedOnlineStatistics, capacity=capacity, value_outlier_std=value_outlier_std)
    )

    for concept_ids, concept_values, concept_value_indicators, units in zip(
        batch["concept_ids"],
        batch["concept_values"] if "concept_values" in batch else batch["number_as_values"],
        batch["concept_value_masks"],
        concept_value_units,
    ):
        for concept_id, concept_value, concept_value_indicator, unit in zip(
            concept_ids, concept_values, concept_value_indicators, units
        ):
            if concept_value_indicator == 1:
                numeric_stats_by_lab[(concept_id, unit)].add(1, concept_value)
    return {"numeric_stats_by_lab": numeric_stats_by_lab}


def agg_statistics(stats1, stats2):
    if stats1.get("numeric_stats_by_lab"):
        for k, v in stats2["numeric_stats_by_lab"].items():
            stats1["numeric_stats_by_lab"][k].combine(v)
    return stats1
