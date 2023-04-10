from flask import Flask, render_template, send_from_directory, redirect
import os
import subprocess
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
import time
from flask import Response, make_response, jsonify

from spacy import displacy

app = Flask(__name__)
FILE_DIR = '../data/ner_data_formatted/txt/'
OTHER_SERVER_PORT = 8888

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



print("Loading pipeline")
nlp = pipeline("ner", model="/anvil/projects/tdm/corporate/battelle-nl/ADE_NER_2023-02-04_4", tokenizer="bert-base-cased")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fileloader')
def files():
    file_list = os.listdir(FILE_DIR)
    return jsonify(file_list)


@app.route('/nlp/<path:filename>')
def download(filename):
    with app.app_context():
        file = filename

        text = ""
        lines = []
        processedlines = []
        print("Starting processing")
        with open(os.path.join("../data/ner_data_formatted/txt/", file)) as f:
            for line in f:
                text += line
                lines.append(line)
                processedlines.append(nlp(line))
        
        print("Processing complete")

        zipped = zip(lines, processedlines)

        #process syllables
        combinedlines = []

        print("<p>Processing syllables</p>")

        currlen = 0
        for item in zipped:
            processed = item[1]
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
                # consider the previous end
                combined.append({"word": word, "entity": processed[i]["entity"], "start": start + currlen, "end": end + currlen})
            currlen += len(item[0])
            combinedlines.extend(combined)
        
        print("<p>Syllables processed</p>")

        for i in range(len(combinedlines)):
            # example output: {'end': None, 'entity': 'LABEL_27', 'index': 131, 'score': 0.9996716, 'start': None, 'word': 'and'}
            combinedlines[i]["label"] = labels[int(combinedlines[i]["entity"].split("_")[1])]
            # if i == 0:
            #     combined[i]["start"] = 0
            #     combined[i]["end"] = len(combined[i]["word"])
            # else:
            #     combined[i]["start"] = combined[i-1]["end"] + 2
            #     combined[i]["end"] = combined[i]["start"] + len(combined[i]["word"])

        # pprint(combinedlines[102:150])
            
        # Generate the visualization using displacy module
        # options = {"ents": labels}

        print("<p>Rendering visualization</p>")

        colors = {}
        # for label in labels:
        #     print(label)
        #     if label[0] == "B":
        #         colors[label] = "rgb(94, 164, 80)"
        #     elif label[0] == "I":
        #         colors[label] = "rgb(152, 177, 207)"
        #     elif label[0] == "L":
        #         colors[label] = "rgb(179, 98, 96)"
        #     elif label[0] == "U":
        #         colors[label] = "rgb(220, 212, 126)"

        for label in labels:
            if label[2:] == "ADE":
                colors[label] = "rgb(94, 164, 80)"
            elif label[2:] == "Dosage":
                colors[label] = "rgb(230, 126, 34)"
            elif label[2:] == "Drug":
                colors[label] = "rgb(52, 152, 219)"
            elif label[2:] == "Duration":
                colors[label] = "rgb(46, 204, 113)"
            elif label[2:] == "Form":
                colors[label] = "rgb(155, 89, 182)"
            elif label[2:] == "Frequency":
                colors[label] = "rgb(241, 196, 15)"
            elif label[2:] == "Reason":
                colors[label] = "rgb(211, 84, 0)"
            elif label[2:] == "Route":
                colors[label] = "rgb(149, 165, 166)"
            elif label[2:] == "Strength":
                colors[label] = "rgb(230, 126, 34)"
        



            

        options = {"ents": [ent for ent in labels if ent != "O"], "colors": colors}

        # print("len of ents")
        # print(len(options["ents"]))
        # doc = {"text": text, "ents": [{"start": i["start"], "end": i["end"], "label": i["label"]} for i in combinedlines]}
        doc = {"text": text, "ents": [{"start": i["start"], "end": i["end"], "label": i["label"]} for i in combinedlines if i["label"] != "O"]}

        html = displacy.render(doc, style="ent", options=options, manual=True)
        
        resp = make_response(html)
        resp.headers['Content-Type'] = 'text/html'
        return resp

@app.route('/streamnlp/<path:filename>')
def newfunction(filename):
    def process1():
        file = filename

        text = ""
        lines = []
        processedlines = []
        yield "<p>Starting processing</p>"
        with open(os.path.join("../data/ner_data_formatted/txt/", file)) as f:
            for line in f:
                text += line
                lines.append(line)
                processedlines.append(nlp(line))
        
        yield "Processing complete"

        zipped = zip(lines, processedlines)

        #process syllables
        combinedlines = []

        yield "<p>Processing syllables</p>"

        currlen = 0
        for item in zipped:
            processed = item[1]
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
                # consider the previous end
                combined.append({"word": word, "entity": processed[i]["entity"], "start": start + currlen, "end": end + currlen})
            currlen += len(item[0])
            combinedlines.extend(combined)
        
        yield "<p>Syllables processed</p>"

        for i in range(len(combinedlines)):
            # example output: {'end': None, 'entity': 'LABEL_27', 'index': 131, 'score': 0.9996716, 'start': None, 'word': 'and'}
            combinedlines[i]["label"] = labels[int(combinedlines[i]["entity"].split("_")[1])]
            # if i == 0:
            #     combined[i]["start"] = 0
            #     combined[i]["end"] = len(combined[i]["word"])
            # else:
            #     combined[i]["start"] = combined[i-1]["end"] + 2
            #     combined[i]["end"] = combined[i]["start"] + len(combined[i]["word"])

        # pprint(combinedlines[102:150])
            
        # Generate the visualization using displacy module
        # options = {"ents": labels}

        yield "<p>Rendering visualization</p>"

        options = {"ents": [ent for ent in labels if ent != "O"]}

        # print("len of ents")
        # print(len(options["ents"]))
        # doc = {"text": text, "ents": [{"start": i["start"], "end": i["end"], "label": i["label"]} for i in combinedlines]}
        doc = {"text": text, "ents": [{"start": i["start"], "end": i["end"], "label": i["label"]} for i in combinedlines if i["label"] != "O"]}

        html = displacy.render(doc, style="ent", options=options, manual=True)
        
        with app.app_context():
            resp = make_response(html)
            resp.headers['Content-Type'] = 'text/html'
            # yield resp
            yield html
    return Response(process1(), mimetype='text/html')


@app.route('/test')
def test():
    def generate():
        yield 'Starting...\n'
        time.sleep(2)
        yield 'Loading step 1...\n'
        time.sleep(2)
        yield 'Loading step 2...\n'
        time.sleep(2)
        yield 'Done!\n'
    Response(generate(), mimetype='text')

if __name__ == '__main__':
    app.run(debug=True, port=8000)
