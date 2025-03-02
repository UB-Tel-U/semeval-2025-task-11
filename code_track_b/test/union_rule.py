import pandas as pd
import numpy as np
import os

model1_path = "../public_data_test/track_b/pred_test_xlmroberta/" 
model2_path = "../public_data_test/track_b/pred_test_distilroberta/" 
model3_path = "../public_data_test/track_b/pred_test_illama3/" 

for csv_file in os.listdir(model1_path):
    if csv_file.endswith(".csv"):
        # Load three CSV prediction files
        csv_files = [model1_path+csv_file, model2_path+csv_file, model3_path+csv_file]
        
        # Read all CSV files into DataFrames
        dfs = [pd.read_csv(file) for file in csv_files]
        target_labels = dfs[0].columns
        target_labels = [col for col in target_labels if col not in ['id', 'text']]

        # Ensure all DataFrames have the same order of samples
        for i in range(1, len(dfs)):
            assert dfs[i]["id"].equals(dfs[0]["id"]), "Mismatch in row ordering across CSVs"

        # Combine predictions: Any label marked as '1' in any model remains '1'
        ensemble_predictions = (sum(df[target_labels] for df in dfs) > 0).astype(int)

        # Keep 'id' column from the first dataframe
        ensemble_df = dfs[0][["id"]].copy()
        ensemble_df[target_labels] = ensemble_predictions

        # Save the ensembled results to a new CSV file
        ensemble_df.to_csv(f'../public_data_test/track_b/pred_test_unionrule/{csv_file}', index=False)
