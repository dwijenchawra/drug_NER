import os
import json
import sys
# sys.path.append("../")
# sys.path.append(os.getcwd())
from argparse import ArgumentParser
import numpy as np

import torch
from datasets import Dataset
from transformers import DataCollatorForTokenClassification, BertForTokenClassification, BertTokenizer, DefaultDataCollator
from transformers import TrainingArguments, Trainer
from bert.ner.utilities import *

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

def main():
    
    ray.init(ignore_reinit_error=True, num_cpus=12)

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
    train_data_path = os.path.join(data_path, "train.tsv")
    test_data_path = os.path.join(data_path, "test.tsv")
    raw_train_data = load_data(train_data_path)
    raw_test_data = load_data(test_data_path)
    raw_train_data = [(i, j) for i, j in raw_train_data if len(i) > 2 and not all(k=="O" for k in j)]
    raw_test_data = [(i, j) for i, j in raw_test_data if len(i) > 2 and not all(k=="O" for k in j)]

    # Tokenize data
    train_data = [tokenize_with_labels(tokenizer, i, j, '[PAD]') for i, j in raw_train_data if len(i) > 2]
    test_data = [tokenize_with_labels(tokenizer, i, j, '[PAD]') for i, j in raw_test_data if len(i) > 2]
    train_sents, train_labels = zip(*train_data)
    test_sents, test_labels = zip(*test_data)
    train_labels = remove_bad_labels(train_labels, config["bad_labels"])

    # Format Model inputs
    MAX_LEN = train_params["max_len"]
    train_inputs = [tokenizer.convert_tokens_to_ids(i) for i in train_sents]
    # train_inputs = pad_sequences(train_inputs, maxlen=MAX_LEN, dtype="long",
    #                              value=0.0, truncating="post", padding="post")
    input_pad = tokenizer._convert_token_to_id('[PAD]')
    train_inputs = np.array([pad_sequence(i, value=input_pad) for i in train_inputs], dtype='int32')

    # Format and save tags
    os.makedirs(model_params["out_folder"], exist_ok=True)
    tag_values = sorted(list(set([j for i in train_labels for j in i])))
    tag2idx = {j: i for i, j in enumerate(tag_values)}
    tag2idx["[PAD]"] = -100
    tag2idx_file = os.path.join(model_params["out_folder"], "tag2idx.json")

    with open(tag2idx_file, 'w') as tag_file:
        json.dump(tag2idx, tag_file)
        tag_file.close()

    train_tags = [[tag2idx[j] for j in i] for i in train_labels]
    # train_tags = pad_sequences(train_tags, maxlen=MAX_LEN, value=tag2idx["PAD"],
    #                            dtype="long", padding="post", truncating="post")
    train_tags = np.array([pad_sequence(i, value=tag2idx['[PAD]']) for i in train_tags], dtype='int32')
    attention_masks = np.array([[int(j != 0) for j in i] for i in train_inputs])

    raw_train_tags = [j for i, j in raw_train_data]
    train_tag_counts = get_entity_type_counts(raw_train_tags, tag2idx)
    train_data, val_data = get_train_test_split_multi_label(
        train_inputs, train_tags, attention_masks, test_size=0.1, stratify=train_tag_counts
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
        fp16=True,
        per_device_train_batch_size=train_params["batch_size"],
        per_device_eval_batch_size=train_params["batch_size"],
        num_train_epochs=train_params["num_epochs"],
        load_best_model_at_end=True,
        report_to="tensorboard",
        metric_for_best_model="f1",
        greater_is_better=True 
        )
    
    def get_model():
        model = BertForTokenClassification.from_pretrained(model_params["pretrain_path"], num_labels=len(tag2idx))
        return model
    
    # data_collator = DataCollatorForTokenClassification(tokenizer)
    data_collator = DefaultDataCollator()
    trainer = Trainer(
        model_init=get_model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda x: compute_metrics(x, tag2idx)
        )
    
    tune_config = {
        "per_device_train_batch_size": tune.choice([16,32]),
        "per_device_eval_batch_size": tune.choice([16,32]),
        "num_train_epochs": 5,
        "weight_decay": tune.loguniform(0.1, 0.0001),
        "learning_rate": tune.uniform(1e-5, 5e-5)
    }
    
    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric='eval_f1',
        mode='max',
    )
    
    reporter = CLIReporter(
        parameter_columns={
            "weight_decay": "w_decay",
            "learning_rate": "lr",
            "per_device_train_batch_size": "train_bs/gpu",
            "num_train_epochs": "num_epochs",
        },
        metric_columns=["eval_f1", "eval_acc", "eval_loss", "epoch", "training_iteration"],
    )
    
    best_run = trainer.hyperparameter_search(
        hp_space=lambda _: tune_config,
        backend='ray',
        n_trials=16,
        direction='maximize',
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir="$SCRATCH/n2c2/ray_results/",
        name="tune_transformer_hb",
        log_to_file=True,
    )
    
    print("BEST HYPERPARAMETERS:")
    print(best_run.hyperparameters)
    
    for k,v in best_run.hyperparameters.items():
        setattr(train_args, k, v)

    trainer.train()
    trainer.evaluate()
    trainer.save_model(model_params["out_folder"])

if __name__ == "__main__":
    main()
