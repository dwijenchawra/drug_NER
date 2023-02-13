import os
import json
from argparse import ArgumentParser
import time

import torch
from datasets import Dataset
from transformers import DataCollatorForTokenClassification, BertForTokenClassification, BertTokenizer 
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import deepspeed
from ner_utilities import *

os.environ["MASTER_ADDR"] = "localhost"
# os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
# os.environ["RANK"] = "0"
# os.environ["LOCAL_RANK"] = "0"
# os.environ["WORLD_SIZE"] = "1"

# for torchrun
# local_rank = int(os.environ["LOCAL_RANK"])
# print("localrank -- " + str(local_rank))

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

configFilePath = "../properties/biobert_barilla_properties.json"

def main():
    parser = ArgumentParser()
    # parser.add_argument("-config", help="Path to configurations file.")
    # args = parser.parse_args()

    parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    cmd_args = parser.parse_args()



    with open(configFilePath) as config_file:
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
    train_data_path = os.path.join(data_path, "train_data.tsv")
    test_data_path = os.path.join(data_path, "test_data.tsv")
    train_data = load_data(train_data_path)
    test_data = load_data(test_data_path)

    # Tokenize data
    train_data = [tokenize_with_labels(tokenizer, i, j, '[PAD]') for i, j in train_data if len(i) > 2]
    test_data = [tokenize_with_labels(tokenizer, i, j, '[PAD]') for i, j in test_data if len(i) > 2]
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

    # Split into Train and Validation for training
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
        # logging_steps=train_params["ckpt_steps"],
        # save_steps=train_params["ckpt_steps"],
        learning_rate=train_params["learning_rate"],
        weight_decay=train_params["weight_decay"],
        per_device_train_batch_size=train_params["batch_size"],
        per_device_eval_batch_size=train_params["batch_size"],
        num_train_epochs=train_params["num_epochs"],
        metric_for_best_model="loss",
        load_best_model_at_end=True
        # deepspeed="deepspeed_config.json"
        )
    
    model = BertForTokenClassification.from_pretrained(model_params["pretrain_path"], num_labels=len(tag2idx))

    # model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
    model = torch.nn.parallel.DistrubutedDataParallel(model, device_ids=[0,1,2,3], find_unused_parameters=False)

    # print("isparallelizable   --    " + str(model.is_parallelizable))
    data_collator = DataCollatorForTokenClassification(tokenizer)
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
