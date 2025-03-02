import pandas as pd
import os
import itertools
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Define models with a short name and their prediction directory
models = [
    ("xlm-roberta", "public_data_test/track_c/pred_dev_baseline_preprocessing_augmentation/FacebookAI/xlm-roberta-base/"),    
    ("distilroberta", "public_data_test/track_c/pred_dev_baseline_preprocessing_augmentation/j-hartmann/emotion-english-distilroberta-base/"),
    ("illama3", "public_data_test/track_c/pred_dev_llama3/"),
    ("bert", "public_data_test/track_c/pred_dev_baseline_preprocessing_augmentation/bert-base-multilingual-cased/"),
    ("distilbert", "public_data_test/track_c/pred_dev_baseline_preprocessing_augmentation/distilbert-base-multilingual-cased/"),
]

def get_average_results(metric, evaluation_results):
    return sum(d[metric] for d in evaluation_results) / len(evaluation_results)

all_evaluation_results = []

# Iterate over all combinations of 2 and 3 models
for r in range(2, len(models) + 1):
    for combo in itertools.combinations(models, r):
        # Unpack model names and paths for the current combination
        combo_names = [name for name, path in combo]
        combo_paths = [path for name, path in combo]
        print(f"Processing combination: {combo_names}")

        # Create an output directory and filename that includes the combination of model names
        combo_str = "_".join(combo_names)
        pred_path = os.path.join("public_data_test/track_c/pred_dev_union/", f"{combo_str}/")
        os.makedirs(pred_path, exist_ok=True)

        gold_path = "public_data_test/track_c/dev/"
        evaluation_results = []
        # Use the first model's directory as the reference for available CSV files
        ref_path = combo_paths[0]
        for csv_file in os.listdir(ref_path):
            if csv_file.endswith(".csv"):
                # Build full file paths for each model in the combination
                csv_files = []
                missing_file = False
                for model_path in combo_paths:
                    file_path = os.path.join(model_path, csv_file)
                    if os.path.exists(file_path):
                        csv_files.append(file_path)
                    else:
                        print(f"File {csv_file} not found in {model_path}. Skipping this file for combination {combo_names}.")
                        missing_file = True
                        break
                if missing_file:
                    continue

                # Read CSV files into DataFrames
                dfs = [pd.read_csv(file) for file in csv_files]

                # Determine target labels (exclude 'id' and 'text' if present)
                target_labels = [col for col in dfs[0].columns if col not in ['id', 'text']]

                # Ensure all DataFrames have the same order of samples using the 'id' column
                for df in dfs[1:]:
                    assert df["id"].equals(dfs[0]["id"]), f"Mismatch in row ordering for file {csv_file} in combination {combo_names}"

                # For union rule, set the threshold to 1: if any model predicts positive, the union is positive
                threshold = 1

                # Ensemble by summing predictions and applying the threshold
                ensemble_predictions = (sum(df[target_labels] for df in dfs) >= threshold).astype(int)

                # Build the ensemble DataFrame, keeping the 'id' column from the reference DataFrame
                ensemble_df = dfs[0][["id"]].copy()
                ensemble_df[target_labels] = ensemble_predictions

                output_file = os.path.join(pred_path, csv_file)
                ensemble_df.to_csv(output_file, index=False)
                print(f"Saved ensemble for combination {combo_names} to {output_file}")

                # Load the gold standard and predicted datasets.
                # The gold standard file is assumed to be named by removing the "pred_" prefix from csv_file.
                gold_standard = pd.read_csv(os.path.join(gold_path, csv_file[len("pred_"):]))
                predictions = pd.read_csv(output_file)

                # Drop 'id' and 'text' columns if they exist
                gold_standard.drop(columns=['id', 'text'], errors='ignore', inplace=True)
                predictions.drop(columns=['id', 'text'], errors='ignore', inplace=True)

                # Compute precision, recall, and F1-score (macro, micro & weighted) and accuracy
                precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                    gold_standard, predictions, average="macro", zero_division=0
                )
                precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
                    gold_standard, predictions, average="micro", zero_division=0
                )
                precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
                    gold_standard, predictions, average="weighted", zero_division=0
                )
                acc = accuracy_score(gold_standard, predictions)

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
        
        # Save per-combination evaluation results
        results_file = os.path.join("code_track_c/dev/evaluation/union_rule/", f"{combo_str}.csv")
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        pd.DataFrame(evaluation_results).to_csv(results_file, index=False)

        dict_pred = {
            'model': combo_str,
            'avg_f1_macro': get_average_results("f1_macro", evaluation_results),
            'avg_precision_macro': get_average_results("precision_macro", evaluation_results),
            'avg_recall_macro': get_average_results("recall_macro", evaluation_results),
            'avg_f1_micro': get_average_results("f1_micro", evaluation_results),
            'avg_precision_micro': get_average_results("precision_micro", evaluation_results),
            'avg_recall_micro': get_average_results("recall_micro", evaluation_results),
            'avg_f1_weighted': get_average_results("f1_weighted", evaluation_results),
            'avg_precision_weighted': get_average_results("precision_weighted", evaluation_results),
            'avg_recall_weighted': get_average_results("recall_weighted", evaluation_results),
            'avg_accuracy': get_average_results("accuracy", evaluation_results),
            "process": "union rule",
        }

        all_evaluation_results.append(dict_pred)

results_file = "code_track_c/dev/evaluation/all_eval_results.csv"
os.makedirs(os.path.dirname(results_file), exist_ok=True)

new_data = pd.DataFrame(all_evaluation_results)
if os.path.exists(results_file):
    new_data.to_csv(results_file, mode='a', header=False, index=False)
else:
    new_data.to_csv(results_file, mode='w', header=True, index=False)
