from brat_parser import get_entities_relations_attributes_groups
from pathlib import Path
import re
import nltk
nltk.download('punkt')

def parse_ann_txt(txt, ann):
    # Read and format text file
    with open(txt, 'r') as txt_file:
        txt = txt_file.read()
        txt = txt.replace('\n\n\n', '@@@')
        txt = txt.replace('\n\n', '^^')
        txt = txt.replace('\n', ' ')
    entities, relations, attributes, groups = get_entities_relations_attributes_groups(ann)

    return txt, entities, relations

def map_character_tags(txt, entities):
    #mapping every character with its tag
    char_ann = ['O' for i in range(len(txt))]
    for e in entities.values():
        char_ann[e.start] = 'B-' + e.type
        for i in range(e.start + 1, e.end - 1):
            char_ann[i] = 'I-' + e.type
        char_ann[e.end - 1] = 'L-' + e.type
    return char_ann

def separate_annotated_substrings(txt, char_ann):
    #getting annotated substrings separate from unannotated ones
    i = 0
    j = 0
    annotated = False
    split = []
    while True:
        if j == len(txt):
            split.append((txt[i:j], char_ann[i]))
            break
        
        if annotated:
            if char_ann[j] == 'O':
                split.append((txt[i:j], char_ann[i][2:]))
                i = j
                annotated = False
            elif char_ann[j][0] == 'B':
                split.append((txt[i:j], char_ann[i][2:]))
                i = j
        else:
            if char_ann[j] != 'O':
                split.append((txt[i:j], 'O'))
                i = j
                annotated = True
        j += 1
    return split

def get_labeled_tokens(labeled_strings):
    #splitting into words and adding labels
    labeled_tokens = []
    line = 1
    for s,t in labeled_strings:
        s = s.replace('@@@', '. ')
        s = s.replace('^^', '. ')
        tokens = nltk.word_tokenize(s)
        if t == 'O':
            for token in tokens:
                if token in '.!?':
                    line += 1
                matches = re.findall("^[^a-zA-Z0-9]+$", token)
                if len(matches) == 0:
                    labeled_tokens.append((line, token, 'O'))
        else:
            if len(tokens) == 1:
                matches = re.findall("^[^a-zA-Z0-9]+$", tokens[0])
                if len(matches) == 0:
                    labeled_tokens.append((line, tokens[0], 'U-' + t))
            else:
                cleaned = []
                for tok in tokens:
                    if len(re.findall("^[^a-zA-Z0-9]+$", tok)) == 0:
                        cleaned.append(tok)
                labeled_tokens.append((line, cleaned[0], 'B-' + t))
                for i in range(1, len(cleaned) - 1):
                    labeled_tokens.append((line, cleaned[i], 'I-' + t))
                labeled_tokens.append((line, cleaned[-1], 'L-' + t))
    return labeled_tokens

def write_to_tsv(array, path):
    lines = ['\t'.join([str(j) for j in i]) + '\n' for i in array]
    with open(path, 'w') as out_file:
        out_file.writelines(lines)
