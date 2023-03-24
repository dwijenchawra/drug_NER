import os
import sys
# sys.path.append("../")
sys.path.append(os.getcwd())
import json
from argparse import ArgumentParser

import torch
import numpy as np
from datasets import Dataset
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification, DefaultDataCollator
from sklearn.model_selection import train_test_split
from bert.sr.utilities import *
from bert.ner.utilities import pad_sequence

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
    train_data_path = os.path.join(data_path, "train.json")
    train_data = load_data(train_data_path)
    
    # Tokenize Data
    train_sents = [prepare_sentence(i['tokens'], i['relation_tags']) for i in train_data]
    train_inputs = []
    for sentence in tqdm(train_sents):
        train_inputs.append(tokenize_sentence_to_ids(tokenizer, sentence))
    train_labels = [i['is_related'] for i in train_data]

    # Format model inputs
    MAX_LEN = train_params["max_len"]
    input_pad = tokenizer._convert_token_to_id('[PAD]')
    train_inputs = np.array([pad_sequence(i, value=input_pad, max_len=MAX_LEN) for i in train_inputs], dtype='int32')

    # Add attention masks
    attention_masks = np.array([[int(j != 0) for j in i] for i in train_inputs])

    # Split into train and validation sets
    relation_types = [i['relation_type'] for i in train_data]
    rel2idx = {v: k for k, v in enumerate(sorted(set(relation_types)))}
    relation_types = [rel2idx[i] for i in relation_types]
    train_data, val_data = train_test_split(
        list(zip(train_inputs, train_labels, attention_masks)), test_size=0.1
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
        load_best_model_at_end=True,
        report_to="tensorboard",
        gradient_accumulation_steps=train_params["grad_accumulation_steps"],
        fp16=True
        )
    
    model = BertForSequenceClassification.from_pretrained(model_params["pretrain_path"], num_labels=2)
    data_collator = DefaultDataCollator()
    trainer = Trainer(
        model,
        train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda x: compute_metrics(x)
        )

    trainer.train()
    trainer.evaluate()
    trainer.save_model(model_params["out_folder"])

if __name__ == "__main__":
    main()
