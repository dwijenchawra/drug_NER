# BioBERT NER for CQM Concepts

This is the guide for training and using the BioBERT model for the NER task using CQM concepts. For this, the HuggingFace library for using transformer models has been utilized. To start, please install the dependencies in the `requirements.txt` in the scripts/BioBERT folder. This model was trained using Python 3.9. Next, the pretrained model, train & test data, and a sample proprerties file can be found on Box in the folder [BioBERT_NER](https://battelle.app.box.com/file/1030492796128).

Then the model can be finetuned for the NER task by running the following command:

```
python train_ner_model.py -config ../../path/to/biobert_ner_properties.json
```

There is already a trained model in the same [BioBERT_NER](https://battelle.app.box.com/file/1030492796128) folder on Box entitled `cqm_ner_model_2022-09-23`. This model was trained using an 80/20 Train & Test split on CQM NER data. The test set results are shown below.

| Entity         | Precision   | Recall   | Support   |
|----------------|-------------|----------|-----------|
| Change Concept |        0.67 |     0.59 |     7,052 |
| Health Status​  |        0.69 |     0.67 |     6,411 |
| Population​     |        0.76 |     0.77 |     2,025 |
| Output​         |        0.51 |     0.59 |       843 |
| Utilization​    |        0.55 |     0.52 |     1,432 |

To evaluate a new trained model you can use the command:

```
python train_ner_model.py -config ../../path/to/biobert_ner_properties.json
```

This will run predictions on the `test.tsv` file that is assigned to the `out_folder` key in the properties file and print out evaluation metrics.
