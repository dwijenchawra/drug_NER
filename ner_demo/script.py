# %%
import os
import sys
sys.path.append("../")
import json
from argparse import ArgumentParser
from datasets import Dataset
from transformers import DataCollatorForTokenClassification, BertForTokenClassification, BertTokenizer 
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from pprint import pprint
import pandas as pd

import spacy
from spacy import displacy


def load_data(path):
    data = pd.read_csv(path, header=None, delimiter='\t')
    data.columns = ['sent_id', 'text', 'label']
    data = data.groupby('sent_id').agg(list).reset_index()
    data = [(row.text, row.label) for row in data.itertuples()]
    return data

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

# %%
# raw_train_data = load_data("../data/ner_data_formatted/train.tsv")
# raw_test_data = load_data("../data/ner_data_formatted/test.tsv")
# raw_train_data = [(i, j) for i, j in raw_train_data if len(i) > 2 and not all(k=="O" for k in j)]
# raw_test_data = [(i, j) for i, j in raw_test_data if len(i) > 2 and not all(k=="O" for k in j)]

tokenizer = BertTokenizer(vocab_file="/anvil/projects/tdm/corporate/battelle-nl/ADE_NER_2023-02-04_4/vocab.txt", do_lower_case=False)

# # Tokenize data
# train_data = [tokenize_with_labels(tokenizer, i, j, '[PAD]') for i, j in raw_train_data if len(i) > 2]
# test_data = [tokenize_with_labels(tokenizer, i, j, '[PAD]') for i, j in raw_test_data if len(i) > 2]
# train_sents, train_labels = zip(*train_data)
# test_sents, test_labels = zip(*test_data)

# print("Labels:")
# pprint(set([l for sent in train_labels for l in sent])) 

labels = ['B-ADE',
    'B-Dosage',
    'B-Drug',
    'B-Duration',
    'B-Form',
    'B-Frequency',
    'B-Reason',
    'B-Route',
    'B-Strength',
    'I-ADE',
    'I-Dosage',
    'I-Drug',
    'I-Duration',
    'I-Form',
    'I-Frequency',
    'I-Reason',
    'I-Route',
    'I-Strength',
    'L-ADE',
    'L-Dosage',
    'L-Drug',
    'L-Duration',
    'L-Form',
    'L-Frequency',
    'L-Reason',
    'L-Route',
    'L-Strength',
    'O',
    'U-ADE',
    'U-Dosage',
    'U-Drug',
    'U-Duration',
    'U-Form',
    'U-Frequency',
    'U-Reason',
    'U-Route',
    'U-Strength',
    '[PAD]']

print("Loading test.txt")
text = ""
with open("test.txt") as f:
    text = f.readline()

print("Loading pipeline")
nlp = pipeline("ner", model="/anvil/projects/tdm/corporate/battelle-nl/ADE_NER_2023-02-04_4", tokenizer="bert-base-cased")

print("Running pipeline")
processed = nlp(text)

pprint(processed[0:5])


# %%
# pprint(processed)
processedline = ""

for i in processed:
    i["label"] = labels[int(i["entity"].split("_")[1])]
    if i["label"] != 'O':
        processedline += i["word"] + f" {i['label']} "
    else:
        processedline += i["word"] + " "
    
print(processedline)

# %%
#process syllables
combined = []
for i in range(len(processed)):
    if processed[i]["word"].startswith('##'):
        continue
    # Otherwise, combine it with the next string if it starts with "##"
    word = processed[i]["word"]
    start = processed[i]["start"]
    end = processed[i]["end"]
    for j in range(i+1, len(processed)):
        if processed[j]["word"].startswith('##'):
            word += processed[j]["word"][2:]
            end = processed[j]["end"]
        else:
            break
    combined.append({"word": word, "entity": processed[i]["entity"], "start": start, "end": end})



for i in range(len(combined)):
    # example output: {'end': None, 'entity': 'LABEL_27', 'index': 131, 'score': 0.9996716, 'start': None, 'word': 'and'}
    combined[i]["label"] = labels[int(combined[i]["entity"].split("_")[1])]
    # if i == 0:
    #     combined[i]["start"] = 0
    #     combined[i]["end"] = len(combined[i]["word"])
    # else:
    #     combined[i]["start"] = combined[i-1]["end"] + 2
    #     combined[i]["end"] = combined[i]["start"] + len(combined[i]["word"])

pprint(combined[10:15])
    
# Generate the visualization using displacy module
options = {"ents": [ent for ent in labels if ent != "O"]}
doc = {"text": text, "ents": [{"start": i["start"], "end": i["end"], "label": i["label"]} for i in combined if i["label"] != "O"]}


# %%
displacy.serve(doc, style="ent", options=options, manual=True, port=8888)

# %%



