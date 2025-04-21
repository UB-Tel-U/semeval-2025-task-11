import pandas as pd
import os
import itertools
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

# Define models with a short name and their prediction directory
models = [
    ("xlm-roberta", "public_data_test/track_b/pred_dev_baseline_preprocessing_augmentation/FacebookAI/xlm-roberta-base/"),    
    ("distilroberta", "public_data_test/track_b/pred_dev_baseline_preprocessing_augmentation/j-hartmann/emotion-english-distilroberta-base/"),
    ("illama3", "public_data_test/track_b/pred_dev_llama3/"),
    ("bert", "public_data_test/track_b/pred_dev_baseline_preprocessing_augmentation/bert-base-multilingual-cased/"),
    ("distilbert", "public_data_test/track_b/pred_dev_baseline_preprocessing_augmentation/distilbert-base-multilingual-cased/"),

    
]

def get_average_results(metric,evaluation_results):
    return sum(d[metric] for d in evaluation_results) / len(evaluation_results)
all_evaluation_results = []

# Iterate over all combinations of 2 and 3 models
for r in range(2, len(models) + 1):
    for combo in itertools.combinations(models, r):
        # Unpack model names and paths for the current combination
        combo_names = [name for name, path in combo]
        combo_paths = [path for name, path in combo]
        print(f"Processing combination: {combo_names}")

        # Create an output filename that includes the combination of model names
        combo_str = "_".join(combo_names)
        # Define the output directory and create it if it doesn't exist
        pred_path = os.path.join("public_data_test/track_b/pred_dev_majority/", f"{combo_str}/")
        os.makedirs(pred_path, exist_ok=True)

        gold_path = "public_data_test/track_b/dev/"
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

                # Stack intensity predictions: shape = (n_samples, n_models, n_emotions)
                stacked = np.stack([df[target_labels].values for df in dfs], axis=1)

                # Max intensity across models (strongest signal)
                max_intensity = np.max(stacked, axis=1)  # shape: (n_samples, n_emotions)

                # Calculate the ensemble threshold: for r models, threshold = r // 2 + 1
                threshold = r // 2 + 1

                # Binary mask: whether each label meets the threshold (e.g., union or majority)
                binary_mask = (sum(df[target_labels] for df in dfs) >= threshold).astype(int).values  # shape: (n_samples, n_emotions)

                # Apply mask: keep max intensity only if binary decision is 1; otherwise 0
                ensemble_predictions = (binary_mask * max_intensity).astype(int)

                # Build the ensemble DataFrame, keeping the 'id' column from the reference DataFrame
                ensemble_df = dfs[0][["id"]].copy()
                ensemble_df[target_labels] = ensemble_predictions

                
                output_file = os.path.join(pred_path, f"{csv_file}")
                ensemble_df.to_csv(output_file, index=False)
                print(f"Saved ensemble for combination {combo_names} to {output_file}")

                # Load the gold standard and predicted datasets
                gold_standard = pd.read_csv(gold_path+csv_file[len("pred_"):])  # Replace with actual gold standard file path
                predictions = pd.read_csv(pred_path+csv_file)  # Replace with actual predictions file path

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
        results_file = "code_track_b/dev/evaluation/majority_voting/"+combo_str+".csv"

        os.makedirs(os.path.dirname(results_file), exist_ok=True)  # Ensure the folder exists
        pd.DataFrame(evaluation_results).to_csv(results_file, index=False)

        dict_pred = {
            'model': combo_str,
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
            "process" : "majority voting",
        }

        all_evaluation_results.append(dict_pred)

results_file = "code_track_b/dev/evaluation/all_eval_results.csv"
os.makedirs(os.path.dirname(results_file), exist_ok=True)  # Ensure the folder exists

# Convert the dictionary to a DataFrame
new_data = pd.DataFrame(all_evaluation_results)  # Ensure it's wrapped in a list for row-wise addition

# Check if the file exists to determine whether to write headers
if os.path.exists(results_file):
    new_data.to_csv(results_file, mode='a', header=False, index=False)  # Append without headers
else:
    new_data.to_csv(results_file, mode='w', header=True, index=False)  # Write with headers if new file