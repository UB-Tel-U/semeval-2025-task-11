import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import os
import numpy as np


folder_names = {
    "baseline": True,
    "preprocessing": False,
    "augmentation": True,
}

model_process = "_".join([key for key, value in folder_names.items() if value])

models = ['bert-base-multilingual-cased', 'distilbert-base-multilingual-cased', 'FacebookAI/xlm-roberta-base', 'j-hartmann/emotion-english-distilroberta-base', 'llama3']

def get_average_results(metric,evaluation_results):
    return sum(d[metric] for d in evaluation_results) / len(evaluation_results)
all_evaluation_results = []

for model_name in models:
    CHECKPOINT = model_name
    gold_path = "../../public_data_test/track_b/dev/"
    if model_name == "llama3":
        pred_path = "../../public_data_test/track_b/pred_dev_" + CHECKPOINT
    else:
        pred_path = "../../public_data_test/track_b/pred_dev_" + model_process + "/" + CHECKPOINT 

    if not os.path.exists(pred_path):  # Check if folder does not exist
        continue
    
    evaluation_results = []
    
    for csv_file in os.listdir(gold_path):
        if csv_file.endswith(".csv"):
            # Load the gold standard and predicted datasets
            gold_standard = pd.read_csv(gold_path+csv_file)  # Replace with actual gold standard file path
            predictions = pd.read_csv(pred_path+"/pred_"+csv_file)  # Replace with actual predictions file path

            # Drop 'id' and 'text' columns if they exist
            gold_standard.drop(columns=['id', 'text'], errors='ignore', inplace=True)
            predictions.drop(columns=['id', 'text'], errors='ignore', inplace=True)

            precision_macro_scores = []
            recall_macro_scores = []
            f1_macro_scores = []

            precision_micro_scores = []
            recall_micro_scores = []
            f1_micro_scores = []

            precision_weighted_scores = []
            recall_weighted_scores = []
            f1_weighted_scores = []

            acc_scores = []

            for col in gold_standard.columns:

                # Compute precision, recall, and F1-score (macro, micro & weighted)
                precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(gold_standard[col], predictions[col], average="macro", zero_division=0)
                precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(gold_standard[col], predictions[col], average="micro", zero_division=0)
                precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(gold_standard[col], predictions[col], average="weighted", zero_division=0)
                acc = accuracy_score(gold_standard[col], predictions[col])

                precision_macro_scores.append(precision_macro)
                recall_macro_scores.append(recall_macro)
                f1_macro_scores.append(f1_macro)

                precision_micro_scores.append(precision_micro)
                recall_micro_scores.append(recall_micro)
                f1_micro_scores.append(f1_micro)

                precision_weighted_scores.append(precision_weighted)
                recall_weighted_scores.append(recall_weighted)
                f1_weighted_scores.append(f1_weighted)

                acc_scores.append(acc)

            precision_macro = np.mean(precision_macro_scores)
            recall_macro = np.mean(recall_macro_scores)
            f1_macro = np.mean(f1_macro_scores)

            precision_micro = np.mean(precision_micro_scores)
            recall_micro = np.mean(recall_micro_scores)
            f1_micro = np.mean(f1_micro_scores)

            precision_weighted = np.mean(precision_weighted_scores)
            recall_weighted = np.mean(recall_weighted_scores)
            f1_weighted = np.mean(f1_weighted_scores)

            acc = np.mean(acc_scores)


            dict_pred = {
                'lang': os.path.splitext(csv_file)[0],
                'f1_macro': f1_macro,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_micro': f1_micro,
                'precision_micro': precision_micro,
                'recall_micro': recall_micro,
                'f1_weighted': f1_weighted,
                'precision_weighted': precision_weighted,
                'recall_weighted': recall_weighted,
                'accuracy': acc
            }

            evaluation_results.append(dict_pred)
    
    # Append results to a CSV file
    results_file = "evaluation/"+model_process+"/"+model_name+".csv"

    os.makedirs(os.path.dirname(results_file), exist_ok=True)  # Ensure the folder exists
    pd.DataFrame(evaluation_results).to_csv(results_file, index=False)

    dict_pred = {
        'model': CHECKPOINT,
        'avg_f1_macro': get_average_results("f1_macro",evaluation_results),
        'avg_precision_macro': get_average_results("precision_macro",evaluation_results),
        'avg_recall_macro': get_average_results("recall_macro",evaluation_results),
        'avg_f1_micro': get_average_results("f1_micro",evaluation_results),
        'avg_precision_micro': get_average_results("precision_micro",evaluation_results),
        'avg_recall_micro': get_average_results("recall_micro",evaluation_results),
        'avg_f1_weighted': get_average_results("f1_weighted",evaluation_results),
        'avg_precision_weighted': get_average_results("precision_weighted",evaluation_results),
        'avg_recall_weighted': get_average_results("recall_weighted",evaluation_results),
        'avg_accuracy': get_average_results("accuracy",evaluation_results),
        "process" : model_process,
    }

    all_evaluation_results.append(dict_pred)

results_file = "evaluation/all_eval_results.csv"
os.makedirs(os.path.dirname(results_file), exist_ok=True)  # Ensure the folder exists

# Convert the dictionary to a DataFrame
new_data = pd.DataFrame(all_evaluation_results)  # Ensure it's wrapped in a list for row-wise addition

# Check if the file exists to determine whether to write headers
if os.path.exists(results_file):
    new_data.to_csv(results_file, mode='a', header=False, index=False)  # Append without headers
else:
    new_data.to_csv(results_file, mode='w', header=True, index=False)  # Write with headers if new file