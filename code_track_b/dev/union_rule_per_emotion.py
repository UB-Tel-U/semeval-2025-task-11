import pandas as pd
import os
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

# Only these two models
models = [
    ("xlm-roberta", "public_data_test/track_b/pred_dev_baseline_preprocessing_augmentation/FacebookAI/xlm-roberta-base/"),
    ("illama3", "public_data_test/track_b/pred_dev_llama3/"),
]

combo_names = [name for name, path in models]
combo_paths = [path for name, path in models]
combo_str = "_".join(combo_names)

print(f"Processing combination: {combo_names}")

pred_path = os.path.join("public_data_test/track_b/pred_dev_union/", f"{combo_str}/")
os.makedirs(pred_path, exist_ok=True)

gold_path = "public_data_test/track_b/dev/"
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

        # === Stack all model predictions
        stacked = np.stack([df[target_labels].values for df in dfs], axis=1)  # (n_samples, n_models, n_labels)

        # === Take maximum intensity per emotion across models
        max_intensity = np.max(stacked, axis=1)

        # === Union rule (threshold 1 means if any model predicts 1+, we keep it)
        threshold = 1
        binary_mask = (sum(df[target_labels] for df in dfs) >= threshold).astype(int).values

        # === Final prediction: only keep max intensity where mask = 1
        ensemble_predictions = (binary_mask * max_intensity).astype(int)

        # Save predictions
        ensemble_df = dfs[0][["id"]].copy()
        ensemble_df[target_labels] = ensemble_predictions
        output_file = os.path.join(pred_path, csv_file)
        ensemble_df.to_csv(output_file, index=False)
        print(f"Saved ensemble to {output_file}")

        # === Load gold and binarize both gold and prediction
        gold_standard = pd.read_csv(os.path.join(gold_path, csv_file[len("pred_"):]))
        predictions = pd.read_csv(output_file)
        gold_standard.drop(columns=['id', 'text'], errors='ignore', inplace=True)
        predictions.drop(columns=['id', 'text'], errors='ignore', inplace=True)

        # === Binarize both to 0/1 for classification metrics
        gold_bin = (gold_standard >= 1).astype(int)
        pred_bin = (predictions >= 1).astype(int)

        # === Compute per-label metrics
        per_label_results = []
        for label in gold_bin.columns:
            acc = accuracy_score(gold_bin[label], pred_bin[label])
            p, r, f, _ = precision_recall_fscore_support(
                gold_bin[label], pred_bin[label], average=None, zero_division=0, labels=[0, 1]
            )

            per_label_results.extend([
                {"csv": csv_file, "label": label, "metric": "precision", "value": p[1]},
                {"csv": csv_file, "label": label, "metric": "recall", "value": r[1]},
                {"csv": csv_file, "label": label, "metric": "f1", "value": f[1]},
                {"csv": csv_file, "label": label, "metric": "accuracy", "value": acc},
            ])

        # Save per-label metrics per file
        per_label_df = pd.DataFrame(per_label_results)
        per_label_file = os.path.join("code_track_b/dev/evaluation/union_rule/per_label/", f"{combo_str}_{os.path.splitext(csv_file)[0]}.csv")
        os.makedirs(os.path.dirname(per_label_file), exist_ok=True)
        per_label_df.to_csv(per_label_file, index=False)

        all_label_results.extend(per_label_results)

# === Compute average across all files
all_df = pd.DataFrame(all_label_results)
avg_df = all_df.groupby(["label", "metric"])["value"].mean().reset_index()

summary_file = f"code_track_b/dev/evaluation/union_rule/per_label/{combo_str}_average.csv"
avg_df.to_csv(summary_file, index=False)
print(f"Saved average per-label metrics to {summary_file}")
