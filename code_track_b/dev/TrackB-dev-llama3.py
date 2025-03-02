import os
import requests
import json
import time
import pandas as pd
import emoji

# os.environ["CHATAI_API_KEY"] = "[CHATAI_API_KEY]"

# CHATAI_API_KEY = os.environ.get("CHATAI_API_KEY")

CONFIG_PATH = "config.json"  # Adjust the path if needed
with open(CONFIG_PATH, "r") as config_file:
    config = json.load(config_file)

CHATAI_API_KEY = config.get("CHATAI_API_KEY")

PREFIX_PROMPT = """Classify the given text into one or more emotion categories: Joy, Sadness, Fear, Anger, Surprise, or Disgust. Each emotion should be assigned an intensity score between 0 and 3, where 0 means no intensity and 3 means the highest intensity.

The output format should be a comma-separated list of emotions with their intensity scores in parentheses, e.g., 'Joy(2), Fear(1), Anger(0)'.

Do not include emotions that are not specified. Do not add any reasoning, explanation, or extra sentences.

If the text is not in English, translate it first to English. If a direct English translation is not possible, use an intermediate language before translating to English. The translation can be transitive through multiple languages if necessary."""

def emotion_llama_classifier(message):
    url = 'https://api-llm.ub.ac.id/api/chat/completions'
    headers = {
        'Authorization': f'Bearer {CHATAI_API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': 'llama3.3:latest',
        'messages': [{'role': 'user', 'content': PREFIX_PROMPT + " The text is:" + message}]
    }
    response = requests.post(url, headers=headers, json=payload)
    response_json = response.json()
    
    # Handle potential errors
    if 'choices' not in response_json:
        print("Error: Unexpected API response format", response_json)
        return ""
    
    return response_json['choices'][0].get('message', {}).get('content', '').lower()

emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

def classify_emotions_intensity(tweet):
    results = emotion_llama_classifier(tweet)
    print("Processing tweet:", tweet)
    intensity_scores = {emotion: 0 for emotion in emotion_labels}
    for prediction in results.split(','):
        prediction = prediction.strip()
        if '(' in prediction and ')' in prediction:
            try:
                label, score = prediction.split('(')
                score = int(score.replace(')', '').strip())
                label = label.lower().strip()
                if label in intensity_scores:
                    intensity_scores[label] = score
            except ValueError:
                print("Skipping invalid format:", prediction)
    return intensity_scores

folder_path = "../../public_data_test/track_b/dev"
for file in os.listdir(folder_path):
    if file.endswith(".csv")and file=="chn.csv":
        file_path = os.path.join(folder_path, file)
        print(f"Processing {file}")
        df = pd.read_csv(file_path)
        column_names = df.columns
        df.drop(columns=df.columns[2:], inplace=True)
        df[emotion_labels] = pd.DataFrame(df['text'].apply(lambda tweet: classify_emotions_intensity(tweet)).tolist(), index=df.index).astype(int)

        df = df[column_names]
        df.drop(columns="text", inplace=True)
        output_path = '../../public_data_test/track_b/pred_dev_llama3/'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure the folder exists
        df.to_csv(output_path + file, index=False)
