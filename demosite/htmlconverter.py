import os
import pandas as pd
from transformers import BertTokenizer
from transformers import pipeline
from spacy import displacy
from pprint import pprint
from tqdm import tqdm

FILE_DIR = '../data/ner_data_formatted/txt/'

tokenizer = BertTokenizer(vocab_file="/anvil/scratch/x-dchawra/ner_trained/vocab.txt", do_lower_case=False)

labels = ['B-ADE', 'B-Dosage', 'B-Drug', 'B-Duration', 'B-Form', 'B-Frequency', 'B-Reason', 'B-Route', 'B-Strength',
          'I-ADE', 'I-Dosage', 'I-Drug', 'I-Duration', 'I-Form', 'I-Frequency', 'I-Reason', 'I-Route', 'I-Strength',
          'L-ADE', 'L-Dosage', 'L-Drug', 'L-Duration', 'L-Form', 'L-Frequency', 'L-Reason', 'L-Route', 'L-Strength',
          'O', 'U-ADE', 'U-Dosage', 'U-Drug', 'U-Duration', 'U-Form', 'U-Frequency', 'U-Reason', 'U-Route', 'U-Strength',
          '[PAD]']

print("Loading pipeline")
nlp = pipeline("ner", model="/anvil/scratch/x-dchawra/ner_trained", tokenizer="bert-base-cased")

# Define a function to process a file
def process_file(file_path):
    file = file_path
    text = ""
    lines = []
    processedlines = []
    print("Starting processing")
    with open(os.path.join(file)) as f:
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

    with open("spacyrenders/" + os.path.basename(file_path) + ".new.html", "w") as f:
        f.write(html)

# Process all files in the specified directory
for filename in tqdm(os.listdir(FILE_DIR), desc="Processing files", unit="files"):
    if filename.endswith(".txt"):
        file_path = os.path.join(FILE_DIR, filename)
        print("Processing file: " + file_path)
        process_file(file_path)
        exit()