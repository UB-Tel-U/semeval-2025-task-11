import emoji
import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForSequenceClassification
import numpy as np
import datasets
from datasets import load_dataset, concatenate_datasets

from sklearn.metrics import accuracy_score, f1_score

SEED = 42
torch.manual_seed(SEED)

emoticon_dict = {
    ":)": "happy",
    ":-)": "happy",
    ":D": "very happy",
    ":-D": "very happy",
    ":(": "sad",
    ":-(": "sad",
    ":'(": "crying",
    ":/": "confused",
    ":-/": "confused",
    ";)": "wink",
    ";-)": "wink",
    "<3": "love",
    ":|": "neutral",
}


def replace_emoticons(text):
    """
    Replace emoticons in text with their corresponding meanings.
    """
    for emoticon, meaning in emoticon_dict.items():
        text = text.replace(emoticon, f" {meaning} ")
    return text

def replace_emojis(text):
    """
    Replace emojis in text with their corresponding words.
    """
    return emoji.demojize(text, delimiters=(" ", " "))

def preprocess_text(text):
    """
    Full preprocessing pipeline for texts.
    """
    text = replace_emoticons(text)  # Convert emoticons
    text = replace_emojis(text)  # Convert emojis
    return text

def text_augmentation():
    """
    Augment text by replacing words with their synonyms.
    """
    arabic_data = load_dataset("SemEvalWorkshop/sem_eval_2018_task_1", 'subtask5.arabic')
    english_data = load_dataset("SemEvalWorkshop/sem_eval_2018_task_1", 'subtask5.english')
    spanish_data = load_dataset("SemEvalWorkshop/sem_eval_2018_task_1", 'subtask5.spanish')
    arabic_combined = concatenate_datasets([
        arabic_data['train'], 
        arabic_data['validation'], 
        arabic_data['test']
    ])
    english_combined = concatenate_datasets([
        english_data['train'], 
        english_data['validation'], 
        english_data['test']
    ])
    spanish_combined = concatenate_datasets([
        spanish_data['train'], 
        spanish_data['validation'], 
        spanish_data['test']
    ])

    del arabic_data, english_data, spanish_data

    arabic_df = pd.DataFrame(data=arabic_combined)
    english_df = pd.DataFrame(data=english_combined)
    spanish_df = pd.DataFrame(data=spanish_combined)

    del arabic_combined, english_combined, spanish_combined

    aug_df = pd.concat([arabic_df, english_df, spanish_df], ignore_index=True)

    del english_df, arabic_df, spanish_df

    columns_to_keep = ['Tweet', 'anger', 'fear', 'joy', 'sadness', 'surprise', 'disgust']
    aug_df = aug_df[columns_to_keep]

    aug_df.rename(columns={'Tweet': 'text'}, inplace=True)

    for col in ['anger', 'fear', 'joy', 'sadness', 'surprise', 'disgust']:
        aug_df[col] = aug_df[col].astype(int)
    
    return aug_df

train_folder_path = "public_data_test/track_a/train"
dev_folder_path = "public_data_test/track_a/dev"

def load_and_process_folder(folder_path):
    dataframes = []

    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            df.columns = [col.lower() for col in df.columns]
            df["text"] = df.apply(lambda row: preprocess_text(row["text"]), axis=1)
            dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)

    if len(combined_df.columns) >= 6:
        for col in combined_df.columns[-6:]:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').fillna(0).astype(int)

    df = combined_df.iloc[:,1:]
    return df

df_train = load_and_process_folder(train_folder_path)
df_train = pd.concat([df_train, text_augmentation()], ignore_index=True)
df_test = load_and_process_folder(dev_folder_path)

train_dataset = Dataset.from_pandas(df_train.reset_index(drop=True))
test_dataset = Dataset.from_pandas(df_test.reset_index(drop=True))

dataset_dict = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

ID2LABEL = {}
LABEL2ID = {}

for idx,label in enumerate(df_train.columns):
    if label in ['text'] or label in ['id']:
        continue

    ID2LABEL[idx-1] = label
    LABEL2ID[label] = idx-1

def preprocess(batch):
    # rename column
    # batch['ID'] = batch['id']
    batch['Tweet'] = batch['text']

    # get one-hot encoded labels for each example in batch
    # for example: anger and sadness = vector of [1,0,0,0,1]
    batch['labels'] = [[float(batch[label][i]) for label in LABEL2ID] for i in range(len(batch['Tweet']))]
    return batch

preprocessed_datasets = dataset_dict.map(preprocess, batched=True, remove_columns=dataset_dict['train'].column_names)

# j-hartmann/emotion-english-distilroberta-base
CHECKPOINT = 'FacebookAI/xlm-roberta-base'
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
tokenized_datasets = preprocessed_datasets.map(lambda batch: tokenizer(batch['Tweet'], padding="max_length", truncation=True, max_length=512), batched=True, remove_columns=['Tweet'])

model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, problem_type='multi_label_classification', num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True)

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

trainer.train()
trainer.evaluate(tokenized_datasets['test'])
trainer.save_model("model")