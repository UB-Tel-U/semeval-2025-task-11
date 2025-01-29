import pandas as pd
import os

folder_path = "public_data_test/track_a/train"
dataframes = []


for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df.columns = [col.lower() for col in df.columns]
        dataframes.append(df)


combined_df = pd.concat(dataframes, ignore_index=True)

if len(combined_df.columns) >= 6:
    for col in combined_df.columns[-6:]:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').fillna(0).astype(int)

df = combined_df.iloc[:,1:]

ID2LABEL = {}
LABEL2ID = {}

for idx,label in enumerate(df.columns):
    if label in ['text'] or label in ['id']:
        continue

    ID2LABEL[idx-1] = label
    LABEL2ID[label] = idx-1

print(f"ID2LABEL: {ID2LABEL}")
print(f"LABEL2ID: {LABEL2ID}")


from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

# Combine them into a DatasetDict
dataset_dict = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

# Print information about the DatasetDict
print(dataset_dict)

# get emotion counts by split type
split_types = list(dataset_dict.keys())
emotion_split_counts = {}

for label in LABEL2ID:
    for split_type in split_types:
        if label not in emotion_split_counts:
            emotion_split_counts[label] = []
        emotion_split_counts[label].append(sum(dataset_dict[split_type][label]))

print(f"SPLIT_TYPES: {split_types}")
print(f"EMOTION_SPLIT_COUNTS: {emotion_split_counts}")

def preprocess(batch):
    # rename column
    # batch['ID'] = batch['id']
    batch['Tweet'] = batch['text']

    # get one-hot encoded labels for each example in batch
    # for example: anger and sadness = vector of [1,0,0,0,1]
    batch['labels'] = [[float(batch[label][i]) for label in LABEL2ID] for i in range(len(batch['Tweet']))]
    return batch

preprocessed_datasets = dataset_dict.map(preprocess, batched=True, remove_columns=dataset_dict['train'].column_names)
preprocessed_datasets

from transformers import AutoTokenizer

CHECKPOINT = 'bert-base-multilingual-cased'
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
tokenized_datasets = preprocessed_datasets.map(lambda batch: tokenizer(batch['Tweet'], padding="max_length", truncation=True, max_length=512), batched=True, remove_columns=['Tweet'])


# set seed for reproducibility
import torch

SEED = 42
torch.manual_seed(SEED)

# let's clone a model and finetune as a multi-label classification problem
from transformers import AutoModelForSequenceClassification

# source: https://huggingface.co/distilbert-base-uncased
model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, problem_type='multi_label_classification', num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID)

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# this function calculates accuracy per label in a prediction instead of per prediction
def samples_accuracy_score(y_true, y_pred):
    return np.sum(y_true==y_pred) / y_true.size

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    # we sigmoid all logits for multilabel metrics
    predictions = torch.nn.functional.sigmoid(torch.Tensor(logits))
    # we set threshold to 0.50 to classify positive >= 0.50 and negative < 0.50
    predictions = (predictions >= 0.50).int().numpy()
    # overall accuracy measures accuracy of each true label list and prediction list
    overall_accuracy = accuracy_score(labels, predictions)
    # sample accuracy measures accuracy of each true label in a true label list and prediction in prediction list
    samples_accuracy = samples_accuracy_score(labels, predictions)
    # overall f1 measures macro f1 of each true label list and prediction list, ignoring zero division warnings
    overall_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    # samples f1 measures f1 of each true label in a true label list and prediction in prediction list, ignoring zero division warnings
    samples_f1 = f1_score(labels, predictions, average='samples', zero_division=0)
    return {
        'overall_accuracy': overall_accuracy,
        'samples_accuracy': samples_accuracy,
        'overall_f1': overall_f1,
        'samples_f1': samples_f1,
    }

from transformers import TrainingArguments

training_args = TrainingArguments(
    seed=SEED,                          # seed for reproducibility
    output_dir='results',               # output directory to store epoch checkpoints
    num_train_epochs=5,                 # number of training epochs
    optim='adamw_torch',                # default optimizer as AdamW
    per_device_train_batch_size=32,     # 32 train batch size to speed up training
    per_device_eval_batch_size=32,      # 32 eval batch size to speed up evaluation
    evaluation_strategy='epoch',        # set evaluation strategy to each epoch instead of default 500 steps
    save_strategy='epoch',              # set saving of model strategy to each epoch instead of default 500 steps
    load_best_model_at_end=True,        # load the best model with lowest validation loss
    report_to='none',                   # suppress third-party logging
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# let's see what an unfine-tuned bert can do
trainer.train()
trainer.evaluate(tokenized_datasets['test'])
trainer.save_model("model")