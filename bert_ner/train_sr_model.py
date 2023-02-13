import os
import json
from argparse import ArgumentParser

import torch
import transformers
from datasets import Dataset
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification, DefaultDataCollator
from sklearn.model_selection import train_test_split
from sr_utilities import *
from ner_utilities import pad_sequence

def main():

    parser = ArgumentParser()
    parser.add_argument("-config", help="Path to configurations file.")
    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
        config_file.close()
    model_params = config["model_params"]
    train_params = config["train_params"]
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Tokenizer
    vocab_file = os.path.join(model_params["pretrain_path"], "vocab.txt")
    tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=False)

    # Load Data
    data_path = train_params["data_dir"]
    train_data_path = os.path.join(data_path, "train_data.json")
    # test_data_path = os.path.join(data_path, "test_data.json")
    train_data = load_data(train_data_path)
    # test_data = load_data(test_data_path)

    # Tokenize Data
    train_sents = [prepare_sentence(i['tokens'], i['positions']) for i in train_data]
    # test_sents = [prepare_sentence(i['tokens'], i['positions']) for i in test_data]
    train_labels = [i['relation'] for i in train_data]
    # test_labels = [i['relation'] for i in test_data]

    # Format model inputs
    MAX_LEN = train_params["max_len"]
    train_inputs = [tokenizer.convert_tokens_to_ids(i) for i in train_sents]
    input_pad = tokenizer._convert_token_to_id('[PAD]')
    train_inputs = np.array([pad_sequence(i, value=input_pad) for i in train_inputs], dtype='int32')

    # Format labels
    tag_values = sorted(list(set(train_labels)))
    tag2idx = {j: i for i, j in enumerate(tag_values)}
    tag2idx_file = os.path.join(model_params["out_folder"], "tag2idx.json")
    train_tags = [tag2idx(i) for i in train_labels]

    with open(tag2idx_file, 'w') as tag_file:
        json.dump(tag2idx, tag_file)
        tag_file.close()
    
    # Add attention masks
    attention_masks = np.array([[int(j != 0) for j in i] for i in train_inputs])

    # Split into train and validation set
    train_data, val_data = train_test_split(
        list(zip(train_inputs, train_tags, attention_masks)), test_size=0.1
        )
    train_inputs, train_tags, train_masks = zip(*train_data)
    val_inputs, val_tags, val_masks = zip(*val_data)
    train_dataset = Dataset.from_dict({'input_ids': train_inputs, 'labels': train_tags, 'attention_mask': train_masks}).with_format("torch")
    val_dataset = Dataset.from_dict({'input_ids': val_inputs, 'labels': val_tags, 'attention_mask': val_masks}).with_format("torch")

    # Train the model
    train_args = TrainingArguments(
        model_params["out_folder"],
        overwrite_output_dir=True,
        evaluation_strategy='steps',
        save_strategy='steps',
        save_total_limit=2,
        dataloader_drop_last=True,
        logging_steps=train_params["ckpt_steps"],
        save_steps=train_params["ckpt_steps"],
        learning_rate=train_params["learning_rate"],
        weight_decay=train_params["weight_decay"],
        per_device_train_batch_size=train_params["batch_size"],
        per_device_eval_batch_size=train_params["batch_size"],
        num_train_epochs=train_params["num_epochs"],
        load_best_model_at_end=True
        )
    
    model = BertForSequenceClassification.from_pretrained(model_params["pretrain_path"], num_labels=len(tag2idx))
    data_collator = DefaultDataCollator(tokenizer)
    trainer = Trainer(
        model,
        train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda x: compute_metrics(x, tag2idx)
        )

    trainer.train()
    trainer.evaluate()
    trainer.save_model(model_params["out_folder"])

if __name__ == "__main__":
    main()
0