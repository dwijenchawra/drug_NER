import os
import sys
# sys.path.append('../')
import json
from argparse import ArgumentParser
from brat.ner.utilities import parse_ann_txt
from brat.sr.utilities import *
from tqdm import tqdm

def main():
    parser = ArgumentParser()
    parser.add_argument("-data_folder", help="Path to folder containing brat annotation files.")
    parser.add_argument("-out_file", help="File to output results to.")
    args = parser.parse_args()

    text_files = [os.path.join(args.data_folder, i)
                  for i in os.listdir(args.data_folder) if i.endswith(".txt")]
    ann_files = ["".join(i.split(".")[:-1]) + ".ann" for i in text_files]

    annot_data = []
    for txt, ann in tqdm(zip(text_files, ann_files), total=len(text_files)):
        assert os.path.exists(ann), f"ANN file does not exist: {ann}"
        text, entities, relations = parse_ann_txt(txt, ann)
        relation_dict = create_relation_dict(relations)
        char_annots = map_character_tags(text, entities)
        substring_annots = separate_annotated_substrings(text, char_annots)
        sentence_annots = get_labeled_tokens(substring_annots)
        relation_data = get_relation_examples(sentence_annots, relation_dict)
        annot_data.extend(relation_data)
    
    with open(args.out_file, 'w') as out_file:
        json.dump(annot_data, out_file)

if __name__ == "__main__":
    main()