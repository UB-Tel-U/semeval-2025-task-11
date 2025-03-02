import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, classification_report
import os
import tqdm

gold_path = "public_data_test/track_a/dev/"
pred_path = "public_data_test/track_a/pred_dev/"
for csv_file in tqdm(os.listdir(gold_path), desc="Evaluating CSV files"):
    if csv_file.endswith(".csv"):
        # Load the gold standard and predicted datasets
        gold_standard = pd.read_csv("gold_path"+csv_file)  # Replace with actual gold standard file path
        predictions = pd.read_csv("pred_path"+csv_file)  # Replace with actual predictions file path

        # Drop 'id' and 'text' columns if they exist
        gold_standard.drop(columns=['id', 'text'], errors='ignore', inplace=True)
        predictions.drop(columns=['id', 'text'], errors='ignore', inplace=True)

        # Compute precision, recall, and F1-score (macro, micro & weighted)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(gold_standard, predictions, average="macro")
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(gold_standard, predictions, average="micro")
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(gold_standard, predictions, average="weighted")

        # # Print the classification report
        # print("Classification Report:\n")
        # print(classification_report(gold_standard, predictions, target_names=label_columns))

        # # Print macro, micro, and weighted scores
        # print(f"Precision (Macro): {precision_macro:.4f}")
        # print(f"Recall (Macro): {recall_macro:.4f}")
        # print(f"F1-Score (Macro): {f1_macro:.4f}\n")

        # print(f"Precision (Micro): {precision_micro:.4f}")
        # print(f"Recall (Micro): {recall_micro:.4f}")
        # print(f"F1-Score (Micro): {f1_micro:.4f}\n")

        # print(f"Precision (Weighted): {precision_weighted:.4f}")
        # print(f"Recall (Weighted): {recall_weighted:.4f}")
        # print(f"F1-Score (Weighted): {f1_weighted:.4f}")


        dict_pred = {
            'lang': os.path.splitext(csv_file)[0],
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted
        }

        # Append results to a CSV file
        results_file = "evaluation/evaluation_results.csv"
        if not os.path.exists(results_file):
            pd.DataFrame([dict_pred]).to_csv(results_file, index=False)
        else:
            pd.DataFrame([dict_pred]).to_csv(results_file, mode='a', header=False, index=False)