import pandas as pd
import os
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

# Only these two models
models = [
    ("bert", "public_data_test/track_c/pred_dev_baseline_preprocessing_augmentation/bert-base-multilingual-cased/"),
    ("xlm-roberta", "public_data_test/track_c/pred_dev_baseline_preprocessing_augmentation/FacebookAI/xlm-roberta-base/"),    
    ("distilroberta", "public_data_test/track_c/pred_dev_baseline_preprocessing_augmentation/j-hartmann/emotion-english-distilroberta-base/"),
]

combo_names = [name for name, path in models]
combo_paths = [path for name, path in models]
combo_str = "_".join(combo_names)

print(f"Processing combination: {combo_names}")

pred_path = os.path.join("public_data_test/track_c/pred_dev_union/", f"{combo_str}/")
os.makedirs(pred_path, exist_ok=True)

gold_path = "public_data_test/track_c/dev/"
all_label_results = []

ref_path = combo_paths[0]
for csv_file in os.listdir(ref_path):
    if csv_file.endswith(".csv"):
        csv_files = []
        missing_file = False
        for model_path in combo_paths:
            file_path = os.path.join(model_path, csv_file)
            if os.path.exists(file_path):
                csv_files.append(file_path)
            else:
                print(f"File {csv_file} not found in {model_path}. Skipping.")
                missing_file = True
                break
        if missing_file:
            continue

        dfs = [pd.read_csv(file) for file in csv_files]
        target_labels = [col for col in dfs[0].columns if col not in ['id', 'text']]

        for df in dfs[1:]:
            assert df["id"].equals(dfs[0]["id"]), f"Mismatch in row ordering for file {csv_file}"

        threshold = 1
        ensemble_predictions = (sum(df[target_labels] for df in dfs) >= threshold).astype(int)

        ensemble_df = dfs[0][["id"]].copy()
        ensemble_df[target_labels] = ensemble_predictions

        output_file = os.path.join(pred_path, csv_file)
        ensemble_df.to_csv(output_file, index=False)
        print(f"Saved ensemble to {output_file}")

        gold_standard = pd.read_csv(os.path.join(gold_path, csv_file[len("pred_"):]))
        predictions = pd.read_csv(output_file)

        gold_standard.drop(columns=['id', 'text'], errors='ignore', inplace=True)
        predictions.drop(columns=['id', 'text'], errors='ignore', inplace=True)

        per_label_results = []

        for label in gold_standard.columns:
            acc = accuracy_score(gold_standard[label], predictions[label])
            p, r, f, _ = precision_recall_fscore_support(
                gold_standard[label], predictions[label], average=None, zero_division=0, labels=[0, 1]
            )

            per_label_results.extend([
                {"csv": csv_file, "label": label, "metric": "precision", "value": p[1]},
                {"csv": csv_file, "label": label, "metric": "recall", "value": r[1]},
                {"csv": csv_file, "label": label, "metric": "f1", "value": f[1]},
                {"csv": csv_file, "label": label, "metric": "accuracy", "value": acc},
            ])

        # Save per-label results per file
        per_label_df = pd.DataFrame(per_label_results)
        per_label_file = os.path.join("code_track_c/dev/evaluation/union_rule/per_label/", f"{combo_str}_{os.path.splitext(csv_file)[0]}.csv")
        os.makedirs(os.path.dirname(per_label_file), exist_ok=True)
        per_label_df.to_csv(per_label_file, index=False)

        all_label_results.extend(per_label_results)

# === Compute average across all files ===
all_df = pd.DataFrame(all_label_results)
avg_df = all_df.groupby(["label", "metric"])["value"].mean().reset_index()

summary_file = f"code_track_c/dev/evaluation/union_rule/per_label/{combo_str}_average.csv"
avg_df.to_csv(summary_file, index=False)
print(f"Saved average per-label metrics to {summary_file}")
