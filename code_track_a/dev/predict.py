from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import os
import pandas as pd
import emoji
from pathlib import Path

SEED = 42
torch.manual_seed(SEED)

folder_names = {
    "baseline": True,
    "preprocessing": False,
    "augmentation": True,
}

model_process = "_".join([key for key, value in folder_names.items() if value])

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

def create_folder(folder_path):
    Path(folder_path).mkdir(parents=True, exist_ok=True)

def tokenize_text(tweet):
    """ Konversi teks ke format tokenized lalu kembali ke string """
    if isinstance(tweet, list):  # Jika sudah tokenized, kembalikan string
        tweet = " ".join(str(t) for t in tweet)

    tokens = tokenizer(tweet, truncation=True, max_length=512, return_tensors="pt")  # Gunakan PyTorch tensor
    return tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)  # Decode ke teks normal

def classify_emotions(tweet):
    # Run the pipeline on each tweet and get the labels
    results = twitter_emotion_multilabel_classifier(tweet)

    # Threshold for classification
    threshold = 0.5

    # Initialize binary labels with 0s for each emotion
    binary_labels = {emotion: 0 for emotion in emotion_labels}
    
    # Process the results from the pipeline
    for prediction in results[0]:
        label = prediction['label'].lower()  # Ensure lowercase to match the emotion labels
        score = prediction['score']

        # If the label score exceeds the threshold, set the corresponding emotion to 1
        if label in binary_labels and score > threshold:
            binary_labels[label] = 1
    # print([binary_labels[emotion] for emotion in emotion_labels])
    # Return the binary labels as a list

    return [binary_labels[emotion] for emotion in emotion_labels]

models = ['bert-base-multilingual-cased', 'distilbert-base-multilingual-cased', 'FacebookAI/xlm-roberta-base', 'j-hartmann/emotion-english-distilroberta-base']


for model_name in models:
    CHECKPOINT = model_name
    emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    # model = AutoModelForSequenceClassification.from_pretrained("model-XLM-Roberta-Aug")
    model_path = "models_"+model_process+"/"+ CHECKPOINT
    if not os.path.exists(model_path):  # Check if folder does not exist
        continue
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    twitter_emotion_multilabel_classifier = pipeline(task='text-classification', model=model, tokenizer=tokenizer, device=torch.cuda.current_device(), top_k=None)


    folder_path = "../../public_data_test/track_a/dev"
    dataframes = []


    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            print(file)
            df = pd.read_csv(file_path)
            column_names = df.columns
            # print("before",df.columns)
            df.drop(columns=df.columns[2:], inplace=True)

            if folder_names['preprocessing']==True:
                df["text"] = df.apply(lambda row: preprocess_text(row["text"]), axis=1)

            df['text'] = df['text'].apply(lambda tweet: tokenize_text(tweet))

            df[emotion_labels] = pd.DataFrame(df['text'].apply(lambda tweet: classify_emotions(tweet)).to_list(), index=df.index)
            df = df[column_names]
            df.drop(columns="text", inplace=True)
            
            output_path = "../../public_data_test/track_a/pred_dev_" + model_process + "/" + CHECKPOINT
            create_folder(output_path)

            # print("after",df.columns)
            df.to_csv(output_path+"/pred_"+file, index=False)