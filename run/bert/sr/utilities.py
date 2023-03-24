import json
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, log_loss
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

def tokenize_sentence_to_ids(tokenizer, sentence):
    tokens = tokenizer.tokenize(sentence)
    # Add special tokens
    if tokens[0] != '[CLS]':
        tokens.insert(0, '[CLS]')
    if tokens[-1] != '[SEP]':
        if tokens[-1] in '.!?;':
            tokens[-1] = '[SEP]'
        else:
            tokens.append('[SEP]')
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return token_ids

def compute_metrics(out):
    predictions, labels = out
    predict_labels = np.argmax(predictions, axis=1)
    metrics = {
        "precision": precision_score(labels, predict_labels),
        "recall": recall_score(labels, predict_labels),
        "f1": f1_score(labels, predict_labels),
        "roc_auc": roc_auc_score(labels, predictions[:, 1]),
        "loss": log_loss(labels, predictions)
        }
    return metrics

def compute_class_metrics(preds, labels, classes):
    unique_classes = sorted(list(set(classes)))
    metrics = []
    for cat in unique_classes:
        class_preds = [(i, j) for i, j, k in zip(preds, labels, classes) if k == cat]
        preds_, labels_ = zip(*class_preds)
        metrics.append(
            {
                "Label": cat,
                "Precision": precision_score(labels_, preds_),
                "Recall": recall_score(labels_, preds_),
                "F1": f1_score(labels_, preds_),
                "Support": len(labels_)
            }
        )
    metrics = pd.DataFrame.from_records(metrics)
    
    
