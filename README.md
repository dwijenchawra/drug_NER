# Bert Info Extraction

## Overview
This repository holds code needed to perform Information Extraction using BERT models. To do this, pretrained Transformer models are loaded and then fine-tuned using the Python HuggingFace library. Documentation can be found here: [HuggingFace Transformers Docs](https://huggingface.co/docs/transformers/index). The following instructions will outline the process of setting up an environment, downloading dependencies, and then traninig & testing a BERT model for the Named Entity Recognition (NER) task.  

## Getting started

First you must create a conda environment (preferably Python 3.9) and install the dependencies needed to this environment. This can be done with the following two statements:

```
conda create -n bert_nlp python=3.9

pip install -r requirements.txt
```
Where `bert_nlp` is the conda environment name and the `requirements.txt` file can be located in this folder.

Next you will need to setup a configurations file that lets the training and evaluation processes know where data and models are to be stored and loaded. An example configuration can entitled `biobert_barilla_properties.json` can be found in the properties folder. Make sure that the structure and keys of your JSON dictionary match this example. Some important values in this file include:

* **pretrain_path**: Path to folder containing pretrained model. For this challenge, it will be called `biobert-base-cased-v1.2` and will be located in the models folder.
* **out_folder**: Folder that your finetuned model is to be saved to.
* **data_dir**: Folder that contains the training and test data. For this challenge, the data are in the data/Barilla_NER folder.

The rest of the parameters can be left the same for the first assignment.

## Training and Validating

Once the necessary files have been located and the configurations file is complete, training and testing can take place. The model can be finetuned for the NER task by navigating to the `bert_ner` folder and running the following command:

```
python train_ner_model.py -config ../../path/to/biobert_ner_properties.json
```

This script will load the data, do preprocessing and tokenization, train the model, and then save it to the `out_folder` specified in the configurations. Once a model has been trained and saved, we can run evaluation by navigating to the `bert_ner` folder and using the following command:

```
python evaluate_ner_model.py -config ../../path/to/biobert_ner_properties.json
```

## Assignment

The purpose of this lab is to familiarize you with training and evaluating BERT models using the HuggingFace library. For this, you will need to complete a few main steps:

1. Setup a conda environment in Python 3.9
2. Install dependencies needed for this code.
3. Setup configuration file with attributes needed for training and evaluation.
4. Run training process and produce a saved model.
5. Evaluate model against the test set.
6. If time permits, try testing different hyper-parameter values such as learning_rate, weight_decay, batch_size, and num_epochs. These values can be changed in the configurations file.

## Pretrain Model, Data, and Other File locations:

1. Create a folder `n2c2` in your scratch directory. Inside that, make a folder `pretrain`.
2. Put the pretrained model `biobert-base-cased-v1.2` in the folder `pretrain` you just made.
3. In your home directory, make a folder `data` (unless you have one already). Inside `data`, make another folder `ner_data_formatted` and put the `train.tsv` and 'test.tsv' files in it. 
4. The trained models will be outputted into `$SCRATCH/n2c2/Models` with the name as specified in the configuration file. 
5. The tensorboard logging data will be outputted to `$SCRATCH/n2c2/ray_results/` with the name as specified in the hyperparameter search function `train_ner_model.py`. You can use `tensorboard --logdir=<log folder>` to view the tensorboard. 

Good luck!
