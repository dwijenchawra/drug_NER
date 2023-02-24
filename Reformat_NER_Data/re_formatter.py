import re
import nltk
nltk.download('punkt')
from brat_parser import get_entities_relations_attributes_groups
from pathlib import Path
import json
import argparse

def re_formatter(ann, txt, out):
    entities, relations, attributes, groups = get_entities_relations_attributes_groups(ann)
    txt = Path(txt).read_text()
    txt = txt.replace('\n\n\n', '@@@')
    txt = txt.replace('\n\n', '^^')
    txt = txt.replace('\n', ' ')
    
    #creating a dictionary with every non drug-drug relation from relations
    relations_per_drug = {}
    for relation in relations.values():
        relations_per_drug[relation.subj] = relation.obj
        
    #mapping every individual character with its label in a new list based off the span of entities    
    char_ann = [('O', '', '') for i in range(len(txt))]
    for e in entities.values():
        char_ann[e.start] = ('B', e.type, e.id)
        for i in range(e.start + 1, e.end - 1):
            char_ann[i] = ('I', e.type, e.id)
        char_ann[e.end - 1] = ('L', e.type, e.id)
    
    #characters with the same labels are connected to form strings and their corresponding labels  
    i = 0
    j = 0
    annotated = False
    split = []
    while True:
        if j == len(txt):
            split.append((txt[i:j], 'O'))
            break
        
        if annotated:
            if char_ann[j][0] == 'O':
                split.append((txt[i:j], char_ann[i][2] + '-' + char_ann[i][1]))
                i = j
                annotated = False
            elif char_ann[j][0] == 'B':
                split.append((txt[i:j], char_ann[i][2] + '-' + char_ann[i][1]))
                i = j
        else:
            if char_ann[j][0] != 'O':
                split.append((txt[i:j], 'O'))
                i = j
                annotated = True
        j += 1
    
    #this part creates a list of tuples stored in sentences - each tuple corresponds to a sentence and
    # contains a list of tokens and a list of labels for each token
    #this time the labels do not have BILOU indicators, instead they have the entity id and tag
    #ex: 'Vicodin' corresponds to 'T296-Drug'
    sentences = [] 
    line = ([],[])
    for s,t in split:
        s = s.replace('@@@', '. ')
        s = s.replace('^^', '. ')
        tokens = nltk.word_tokenize(s)
        if t == 'O':
            for token in tokens:
                if token == '.':
                    sentences.append(line)
                    line = ([],[])
                matches = re.findall("^[^a-zA-Z0-9]+$", token)
                if len(matches) == 0:
                    line[0].append(token)
                    line[1].append('O')
        else:
            if len(tokens) == 1:
                matches = re.findall("^[^a-zA-Z0-9]+$", tokens[0])
                if len(matches) == 0:
                    line[0].append(tokens[0])
                    line[1].append(t)
            else:
                cleaned = []
                for tok in tokens:
                    if len(re.findall("^[^a-zA-Z0-9]+$", tok)) == 0:
                        cleaned.append(tok)
                line[0].append(cleaned[0])
                line[1].append(t)
                for i in range(1, len(cleaned) - 1):
                    line[0].append(cleaned[i])
                    line[1].append(t)
                line[0].append(cleaned[-1])
                line[1].append(t)
    
    #in every sentence the indices of drugs and non drugs tags are stored in a dictionary
    #loops through both dictionaries to get the combination of every non drug-drug and creates a dictionary
    #in the format required
    #using relations_per_drug we know if a non-drug is related to the specified drug
    output = []
    for s in sentences:
        non_drugs = {}
        drugs = {}
        for i in range(len(s[1])):
            label = s[1][i]
            if label != 'O':
                if label.split('-')[1] == 'Drug':
                    if label in drugs.keys():
                        drugs[label].append(i)
                    else:
                        drugs[label] = [i]
                else:
                    if label in non_drugs.keys():
                        non_drugs[label].append(i)
                    else:
                        non_drugs[label] = [i]
        if len(drugs) == 0 or len(non_drugs) == 0:
            continue
        for subj in non_drugs.keys():
            i_sub = non_drugs[subj]
            if subj.split('-')[0] not in relations_per_drug.keys():
                continue
            for obj in drugs.keys():
                i_obj = drugs[obj]
                dict = {}
                dict["tokens"] = s[0]
                relation_tags = ['O' for i in range(len(s[0]))]
                for i in range(len(relation_tags)):
                    if i in i_sub:
                        relation_tags[i] = "ENTITY1"
                    elif i in i_obj:
                        relation_tags[i] = "ENTITY2"
                dict["relation_tags"] = relation_tags
                dict["entity_tags"] = ['O' if (s[1][i] == 'O') else (s[1][i].split('-')[1]) for i in range(len(s[1]))]
                dict["relation_type"] = subj.split('-')[1] + '-' + obj.split('-')[1]
                dict["is_related"] = 1 if (relations_per_drug[subj.split('-')[0]] == obj.split('-')[0]) else 0
                output.append(dict)
     
    with open(out, "w") as final:
        json.dump(output, final)                

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument('filenames', nargs=3)
   args = parser.parse_args()
   re_formatter(args.filenames[0], args.filenames[1], args.filenames[2])