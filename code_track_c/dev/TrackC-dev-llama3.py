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

PREFIX_PROMPT = """Classify emotion from the given text into one or more: Joy, Fear, Anger, Sadness, Disgust, Surprise. 
The output is only the multilabel classes (Joy, Fear, Anger, Sadness, Disgust, Surprise) and separated by a comma.  
Do not use other unspecified classes. Do not output the reasoning statement or unnecessary sentences. 
If you don't know or unsure, translate into English first. 
If you cannot translate directly to English, translate it first to another known language, and from that another known language to English.
The translation can be transitive through more than one language."""

def emotion_llama_classifier(message):
    url = 'https://api-llm.ub.ac.id/api/chat/completions'
    headers = {
        'Authorization': f'Bearer {CHATAI_API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': 'llama3.3:latest',
        'messages': [{'role': 'user', 'content': PREFIX_PROMPT + " The text is: " + message}]
    }
    response = requests.post(url, headers=headers, json=payload)
    response_json = response.json()
    
    # Handle potential errors
    if 'choices' not in response_json:
        print("Error: Unexpected API response format", response_json)
        return []
    
    return [label.strip().lower() for label in response_json['choices'][0].get('message', {}).get('content', '').split(',')]

def classify_emotions(tweet, available_labels):
    labels = emotion_llama_classifier(tweet)
    print("Processing tweet:", tweet)
    return {label: 1 if label in labels else 0 for label in available_labels}  # Only keep relevant labels

folder_path = "../../public_data_test/track_c/dev"
for file in os.listdir(folder_path):
    output_path = '../../public_data_test/track_c/pred_dev_llama3/'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure the folder exists
    output_path = output_path + "pred_" + file
    if file.endswith(".csv") and not os.path.exists(output_path):
        file_path = os.path.join(folder_path, file)
        print(f"Processing {file}")
        df = pd.read_csv(file_path)
        
        # Extract emotion labels from the file (exclude 'id' and 'text')
        available_labels = [col.lower() for col in df.columns if col.lower() not in ["id", "text"]]
        
        if not available_labels:
            print(f"Warning: No emotion labels found in {file}. Skipping.")
            continue  # Skip processing if no emotion labels exist
        
        df.drop(columns=df.columns[2:], inplace=True)  # Keep only 'id' and 'text'
        
        # Apply classification based on available emotion labels
        emotion_results = df['text'].apply(lambda tweet: classify_emotions(tweet, available_labels)).apply(pd.Series)

        # Merge results with the original dataframe
        df = pd.concat([df, emotion_results], axis=1)

        df.drop(columns="text", inplace=True)
        df.to_csv(output_path, index=False)
