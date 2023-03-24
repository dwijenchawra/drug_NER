import os
import sys
# sys.path.append('../')
from argparse import ArgumentParser
from brat.ner.utilities import *
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
    doc_offset = 0
    for txt, ann in tqdm(zip(text_files, ann_files), total=len(text_files)):
        assert os.path.exists(ann), f"ANN file does not exist: {ann}"
        text, entities, _ = parse_ann_txt(txt, ann)
        char_annots = map_character_tags(text, entities)
        substring_annots = separate_annotated_substrings(text, char_annots)
        token_annots = get_labeled_tokens(substring_annots)
        sentence_ids = [int(i) for i, j, k in token_annots]
        token_annots = [(i + doc_offset, j, k) for i, j, k in token_annots]
        max_id = max(sentence_ids)
        doc_offset += max_id
        annot_data.extend(token_annots)
    
    # Write data to TSV
    write_to_tsv(annot_data, args.out_file)

if __name__ == '__main__':
    main()