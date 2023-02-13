import json
import pandas as pd
import numpy as np
import torch
from seqeval.metrics import classification_report
from tqdm import tqdm
from torch.utils.data import DataLoader
from string import punctuation

def format_data(data):
    formatted_data = {}
    for split in data.keys():
        formatted_data[split] = [
            {
                "tokens": tok,
                "positions": pos,
                "entities": ent,
                "relation": rel,
                "sentence_id": int(idx)
            }
            for tok, pos, ent, rel, idx, doc in data[split]
        ]
    return formatted_data

def write_json_data(data, path):
    assert path.endswith('.json'), f"{path} is not a JSON file path."
    with open(path, 'w') as file:
        json.dump(data, file)
        file.close()

def load_data(path):
    with open(path, 'r') as file:
        data = json.load(file)
        file.close()
    return data

def find_sentence_location(sentence, token):
    return [i for i, j in enumerate(sentence) if j == token]

def prepare_sentence(tokens, positions, mask_ent=False):
    
    # Get LHS and RHS entity locations
    ent1_locs = find_sentence_location(positions, 'ENTITY1')
    ent2_locs = find_sentence_location(positions, 'ENTITY2')
    ent1_start = min(ent1_locs)
    ent1_end = max(ent1_locs)
    ent2_start = min(ent2_locs)
    ent2_end = max(ent2_locs)

    # Mark LHS and RHS spans
    span_trans = str.maketrans({"[": "(", "]": ")", "{": "(", "}": ")"})
    tokens = [i.translate(span_trans) for i in tokens]
    tokens[ent1_start] = "[" + tokens[ent1_start]
    tokens[ent1_end] = tokens[ent1_end] + "]"
    tokens[ent2_start] = "{" + tokens[ent2_start]
    tokens[ent2_end] = tokens[ent2_end] + "}"

    # Merge sentence tokens
    if tokens[-1][-1] not in ".;!?":
        tokens[-1] = tokens[-1] + "."
    tokens = " ".join(tokens)
    return tokens

def compute_metrics():
    pass
    