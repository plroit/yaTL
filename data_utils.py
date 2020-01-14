import codecs
import json
from typing import List

import torch


def pad_and_mask(indices: List, max_length, pad_idx=0):
    n_items = len(indices)
    diff = max_length - n_items
    indices = indices + [pad_idx for _ in range(diff)]
    indices = torch.tensor(indices).long()
    active_mask = torch.zeros_like(indices)
    active_mask[:n_items] = 1
    return indices, active_mask


def read_json_samples(path):
    with codecs.open(path) as f:
        for line in f:
            sample = json.loads(line)
            yield sample


def read_all_labels(labels_path):
    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f]
        labels = [lbl for lbl in labels if lbl]
        return labels
