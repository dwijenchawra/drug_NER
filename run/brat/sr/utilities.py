import re
import nltk
# nltk.download('punkt')

def create_relation_dict(relations):
    #creating a dictionary with every non drug-drug relation from relations
    relations_per_entity = {rel.subj: rel.obj for rel in relations.values()}
    return relations_per_entity

def map_character_tags(txt, entities):
    #mapping every individual character with its label in a new list based off the span of entities    
    char_ann = [('O', '', '') for i in range(len(txt))]
    for e in entities.values():
        char_ann[e.start] = ('B', e.type, e.id)
        for i in range(e.start + 1, e.end - 1):
            char_ann[i] = ('I', e.type, e.id)
        char_ann[e.end - 1] = ('L', e.type, e.id)
    return char_ann

def separate_annotated_substrings(txt, char_ann):
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

    return split

def get_labeled_tokens(labeled_strings):
    #this part creates a list of tuples stored in sentences - each tuple corresponds to a sentence and
    # contains a list of tokens and a list of labels for each token
    #this time the labels do not have BILOU indicators, instead they have the entity id and tag
    #ex: 'Vicodin' corresponds to 'T296-Drug'
    sentences = [] 
    line = ([],[])
    for s,t in labeled_strings:
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
    return sentences

def get_relation_examples(sentences, relation_dict):
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
            if subj.split('-')[0] not in relation_dict.keys():
                continue
            for obj in drugs.keys():
                i_obj = drugs[obj]
                rel_data = {}
                rel_data["tokens"] = s[0]
                relation_tags = ['O' for i in range(len(s[0]))]
                for i in range(len(relation_tags)):
                    if i in i_sub:
                        relation_tags[i] = "ENTITY1"
                    elif i in i_obj:
                        relation_tags[i] = "ENTITY2"
                rel_data["relation_tags"] = relation_tags
                rel_data["entity_tags"] = ['O' if (s[1][i] == 'O') else (s[1][i].split('-')[1]) for i in range(len(s[1]))]
                rel_data["relation_type"] = subj.split('-')[1] + '-' + obj.split('-')[1]
                rel_data["is_related"] = 1 if (relation_dict[subj.split('-')[0]] == obj.split('-')[0]) else 0
                output.append(rel_data)
    
    return output
