import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import os
import tqdm

models = ['bert-base-multilingual-cased', 'distilbert-base-multilingual-cased', 'FacebookAI/xlm-roberta-base']

def get_average_results(metric,evaluation_results):
    return sum(d[metric] for d in evaluation_results) / len(evaluation_results)
all_evaluation_results = []

for model_name in models:
    CHECKPOINT = model_name
    gold_path = "../../public_data_test/track_a/dev/"
    pred_path = "../../public_data_test/track_a/pred_dev_"+CHECKPOINT+"/"
    
    evaluation_results = []
    
    for csv_file in os.listdir(gold_path):
        if csv_file.endswith(".csv"):
            # Load the gold standard and predicted datasets
            gold_standard = pd.read_csv(gold_path+csv_file)  # Replace with actual gold standard file path
            predictions = pd.read_csv(pred_path+"pred_"+csv_file)  # Replace with actual predictions file path

            # Drop 'id' and 'text' columns if they exist
            gold_standard.drop(columns=['id', 'text'], errors='ignore', inplace=True)
            predictions.drop(columns=['id', 'text'], errors='ignore', inplace=True)

            # Compute precision, recall, and F1-score (macro, micro & weighted)
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(gold_standard, predictions, average="macro", zero_division=0)
            precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(gold_standard, predictions, average="micro", zero_division=0)
            precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(gold_standard, predictions, average="weighted", zero_division=0)

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
                'accuracy': f1_weighted
            }

            evaluation_results.append(dict_pred)
    
    # Append results to a CSV file
    results_file = "evaluation/eval_results_"+model_name+".csv"
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
    }

    all_evaluation_results.append(dict_pred)
    
results_file = "evaluation/all_eval_results.csv"
os.makedirs(os.path.dirname(results_file), exist_ok=True)  # Ensure the folder exists
pd.DataFrame(all_evaluation_results).to_csv(results_file, index=False)