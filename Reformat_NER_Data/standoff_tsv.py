import argparse
from brat_parser import get_entities_relations_attributes_groups
from pathlib import Path
import re
import nltk
# nltk.download('punkt')

def standoff_to_tsv(ann, txt, out):
    entities, relations, attributes, groups = get_entities_relations_attributes_groups(ann)
    txt = Path(txt).read_text()
    txt = txt.replace('\n\n\n', '@@@')
    txt = txt.replace('\n\n', '^^')
    txt = txt.replace('\n', ' ')
    
    #mapping every character with its tag
    char_ann = ['O' for i in range(len(txt))]
    for e in entities.values():
        char_ann[e.start] = 'B-' + e.type
        for i in range(e.start + 1, e.end - 1):
            char_ann[i] = 'I-' + e.type
        char_ann[e.end - 1] = 'L-' + e.type    
    
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
    
    reg = "^[^a-zA-Z0-9]+$"
        
    #splitting into words and adding labels
    output = []
    line = 1
    for s,t in split:
        s = s.replace('@@@', '. ')
        s = s.replace('^^', '. ')
        tokens = nltk.word_tokenize(s) 
        if t == 'O':
            for token in tokens:
                if token == '.':
                    line += 1
                matches = re.findall("^[^a-zA-Z0-9]+$", token)
                if len(matches) == 0:
                    output.append((line, token, 'O'))

                
        else:
            if len(tokens) == 1:
                matches = re.findall("^[^a-zA-Z0-9]+$", tokens[0])
                if len(matches) == 0:
                    output.append((line, tokens[0], 'U-' + t))
            else:
                cleaned = []
                for tok in tokens:
                    if len(re.findall("^[^a-zA-Z0-9]+$", tok)) == 0:
                        cleaned.append(tok)
                output.append((line, cleaned[0], 'B-' + t))
                for i in range(1, len(cleaned) - 1):
                    output.append((line, cleaned[i], 'I-' + t))
                output.append((line, cleaned[-1], 'L-' + t))

    f = open(out, 'w')
    for data in output:
        f.write(str(data[0]) + '\t' + data[1] + '\t' + data[2] + '\n')
    f.close()
        

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument('filenames', nargs=3)
   args = parser.parse_args()
   standoff_to_tsv(args.filenames[0], args.filenames[1], args.filenames[2])