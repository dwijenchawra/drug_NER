import re
import csv
import pandas as pd
import numpy as np
import torch
from seqeval.metrics import classification_report
from tqdm import tqdm
from torch.utils.data import DataLoader
from deep_utils import stratify_train_test_split_multi_label
from sklearn.model_selection import train_test_split
from collections import Counter

def load_data(path):
    data = pd.read_csv(path, header=None, delimiter='\t')
    data.columns = ['sent_id', 'text', 'label']
    data = data.groupby('sent_id').agg(list).reset_index()
    data = [(row.text, row.label) for row in data.itertuples()]
    return data

def remove_bad_labels(label_data, bad_labels):
    all_labels = set([j for i in label_data for j in i])
    bad_labels = set([i for i in all_labels for j in bad_labels if j in i])
    
    labels_filt = []
    for sent_labels in label_data:
        labels_filt.append(['O' if i in bad_labels else i for i in sent_labels])
    return labels_filt

def format_sent_data(brat_data):
    sent_data = []
    for i, (sentence, labels, _,_,_,_) in enumerate(brat_data):
        assert len(sentence) == len(labels)
        sent_ids = [i] * len(sentence)
        sent_data.extend(zip(sent_ids, sentence, labels))
    return sent_data

def write_data_tsv(data, out_file):
    with open(out_file, 'w', newline='') as file:
        tsv_writer = csv.writer(file, delimiter='\t')
        tsv_writer.writerows(data)
        file.close()

def tokenize_with_labels(tokenizer, sent_words, sent_labels, special_label):
    tok_sent = []
    labels = []
    for word, label in zip(sent_words, sent_labels):
        if type(word) == str:
            tok_word = tokenizer.tokenize(word)
            n_subwords = len(tok_word)

            tok_sent.extend(tok_word)
            labels.extend([label] * n_subwords)
    
    # Add special tokens
    if tok_sent[0] != '[CLS]':
        tok_sent.insert(0, '[CLS]')
        labels.insert(0, special_label)
    if tok_sent[-1] != '[SEP]':
        if tok_sent[-1] not in '.!?;':
            tok_sent.append('.')
            labels.append('O')
        tok_sent.append('[SEP]')
        labels.append(special_label)

    return tok_sent, labels

def compute_metrics(p, tag2idx):
    idx2tag = {v: k for k, v in tag2idx.items()}
    pad_idx = tag2idx['[PAD]']
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [[idx2tag[p] for (p, l) in zip(prediction, label) if l != pad_idx] for prediction, label in zip(predictions, labels)]
    true_labels = [[idx2tag[l] for (p, l) in zip(prediction, label) if l != pad_idx] for prediction, label in zip(predictions, labels)]

    results = classification_report(true_predictions, true_labels, output_dict=True)
    metrics = {
        "precision": results["micro avg"]["precision"],
        "recall": results["micro avg"]["recall"],
        "f1": results["micro avg"]["f1-score"]
        }

    return metrics

def pad_sequence(sentence, max_len=200, value=0):
    if len(sentence) >= max_len:
        return sentence[:max_len]
    else:
        return sentence + [value] * (max_len - len(sentence))

def get_label_predictions(model, eval_dataset, tag2idx, batch_size=1, device="cpu"):
    true_predictions = []
    true_labels = []
    idx2tag = {v: k for k, v in tag2idx.items()}
    pad_idx = tag2idx['[PAD]']
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

    model.eval()
    with torch.no_grad():
        for row in tqdm(eval_dataloader):
            x = row['input_ids'].to(device)
            out = model(x).logits
            preds = np.argmax(out.cpu(), axis=-1).squeeze().tolist()
            labels = row['labels'].squeeze().tolist()

            for pred, label in zip(preds, labels):
                true_predictions.append([idx2tag[p] for p, l in zip(pred, label) if l != pad_idx])
                true_labels.append([idx2tag[l] for p, l in zip(pred, label) if l != pad_idx])
    
    return true_predictions, true_labels

def get_train_test_split_multi_label(x, y, attn_masks, test_size, stratify=None):
    if stratify is None:
        # Split into Train and Test by random sample
        train_data, test_data = train_test_split(
            list(zip(x, y, attn_masks)), test_size=test_size
            )
    else:
        # Split into Train and Test by stratified sample
        x_train, x_test, y_train, y_test = stratify_train_test_split_multi_label(
            x=list(range(len(y))), y=stratify, test_size=test_size
            )
        x_train = x_train.astype(int)
        x_test = x_test.astype(int)
        train_input, train_tags, train_attn = x[x_train], y[x_train], attn_masks[x_train]
        test_input, test_tags, test_attn = x[x_test], y[x_test], attn_masks[x_test]
        train_data = list(zip(train_input, train_tags, train_attn))
        test_data = list(zip(test_input, test_tags, test_attn))
    
    return train_data, test_data

def get_entity_type_counts(tags, tag2idx):
    drop_tags = {'O', '[PAD]'}
    tag_ents = [
        [re.sub(r"^[^-]*-", "", j) for j in i if j.startswith("U-") or j.startswith("B-")]
        for i in tags
        ]
    tag_names = sorted(set(re.sub(r"^[^-]*-", "", i) for i in tag2idx.keys() if i not in drop_tags))
    tag_counts = [Counter(i) for i in tag_ents]
    tag_counts = np.array([[i[j] for j in tag_names] for i in tag_counts])

    return tag_counts
