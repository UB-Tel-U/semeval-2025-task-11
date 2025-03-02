from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
import pandas as pd
import emoji

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
    for emoticon, meaning in emoticon_dict.items():
        text = text.replace(emoticon, f" {meaning} ")
    return text

def replace_emojis(text):
    return emoji.demojize(text, delimiters=(" ", " "))

def preprocess_text(text):
    text = replace_emoticons(text)
    text = replace_emojis(text)
    return text

emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
CHECKPOINT = 'FacebookAI/xlm-roberta-base'
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model = AutoModelForSequenceClassification.from_pretrained("model_xlm-roberta")
classifier = pipeline(task='text-classification', model=model, tokenizer=tokenizer, device=torch.cuda.current_device(), top_k=None)

def tokenize_text(tweet):
    if isinstance(tweet, list):
        tweet = " ".join(str(t) for t in tweet)
    tokens = tokenizer(tweet, truncation=True, max_length=512, return_tensors="pt")
    return tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)

def classify_emotions(tweet):
    results = classifier(tweet)
    intensity_scores = {emotion: 0 for emotion in emotion_labels}
    for prediction in results[0]:
        label = prediction['label'].lower()
        score = prediction['score']
        if label in intensity_scores:
            intensity_scores[label] = round(score * 3)  # Convert to intensity scale (0-3)
    return [intensity_scores[emotion] for emotion in emotion_labels]

folder_path = "../public_data_test/track_b/test"
for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        file_path = os.path.join(folder_path, file)
        print(f"Processing {file}")
        df = pd.read_csv(file_path)
        column_names = df.columns
        df.drop(columns=df.columns[2:], inplace=True)
        df["text"] = df["text"].apply(preprocess_text)
        df[emotion_labels] = pd.DataFrame(df['text'].apply(classify_emotions).tolist(), index=df.index)
        df = df[column_names]
        df.drop(columns="text", inplace=True)
        df.to_csv(f'../public_data_test/track_b/pred_test_xlm-roberta/pred_{file}', index=False)
