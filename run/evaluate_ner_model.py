import os
import sys
sys.path.append("../")
import json
from argparse import ArgumentParser
from datasets import Dataset
from transformers import DataCollatorForTokenClassification, BertForTokenClassification, BertTokenizer 
from bert.ner.utilities import *

def main():

    parser = ArgumentParser()
    parser.add_argument("-config", help="Path to configurations file.")
    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
        config_file.close()
    batch_size = 64
    model_params = config["model_params"]
    train_params = config["train_params"]

    # Setup Tokenizer
    vocab_file = os.path.join(model_params["out_folder"], "vocab.txt")
    tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=False)

    # Load Data
    data_path = train_params["data_dir"]
    test_data_path = os.path.join(data_path, "test.tsv")
    test_data = load_data(test_data_path)
    test_data = [(i, j) for i, j in test_data if len(i) > 2 and not all(k=="O" for k in j)]

    # Tokenize data
    test_data = [tokenize_with_labels(tokenizer, i, j, '[PAD]') for i, j in test_data if len(i) > 2]
    test_sents, test_labels = zip(*test_data)
    test_labels = remove_bad_labels(test_labels, config["bad_labels"])

    # Format model inputs
    MAX_LEN = train_params["max_len"]
    test_inputs = [tokenizer.convert_tokens_to_ids(i) for i in test_sents]
    input_pad = tokenizer._convert_token_to_id('[PAD]')
    test_inputs = np.array([pad_sequence(i, value=input_pad) for i in test_inputs], dtype='int32')

    tag2idx_path = os.path.join(model_params["out_folder"], "tag2idx.json")
    with open(tag2idx_path, 'r') as tag_file:
        tag2idx = json.load(tag_file)
        tag_file.close()
    
    test_tags = [[tag2idx[j] for j in i] for i in test_labels]
    test_tags = np.array([pad_sequence(i, value=tag2idx['[PAD]']) for i in test_tags], dtype='int32')
    attention_masks = [[float(j != 0.0) for j in i] for i in test_inputs]
    test_dataset = Dataset.from_dict({'input_ids': test_inputs, 'labels': test_tags, 'attention_mask': attention_masks}).with_format("torch")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = BertForTokenClassification.from_pretrained(model_params["out_folder"])
    model = model.eval()
    model = model.to(device)
    pred, labels = get_label_predictions(model, test_dataset, tag2idx, batch_size=batch_size, device=device)
    
    # Display Results
    print()
    print(classification_report(pred, labels))

if __name__ == '__main__':
    main()
