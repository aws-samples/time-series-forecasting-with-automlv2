import json
import os
import pathlib
import pandas as pd
from sklearn.metrics import mean_absolute_error
import numpy as np
from math import sqrt

def rmse(y_true, y_pred):
    return np.sqrt(((y_pred - y_true) ** 2).mean())

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_absolute_scaled_error(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mae_baseline = mean_absolute_error(y_true[1:], y_true[:-1])
    return mae / mae_baseline

def weighted_absolute_percentage_error(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100

        
if __name__ == "__main__":
    # Paths to your predicted and true values
    pred_path = "/opt/ml/processing/input/predictions/batch-food-demand.csv.out"
    true_path = "/opt/ml/processing/input/true_labels/test.csv"
    
    pred_df = pd.read_csv(pred_path)
    true_df = pd.read_csv(true_path)
    
    # Merge the DataFrames on specified columns
    merged_df = pd.merge(pred_df, true_df, how='inner', on=["product_code", "location_code", "timestamp"])
    y_true = merged_df["unit_sales"]
    y_pred = merged_df["mean"]
    
    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = rmse(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mase = mean_absolute_scaled_error(y_true, y_pred)
    wape = weighted_absolute_percentage_error(y_true, y_pred)

    # Prepare the evaluation report dictionary with the additional metrics
    report_dict = {
        "forecasting_metrics": {
            "MAE": {"value": mae, "standard_deviation": "NaN"},
            "RMSE": {"value": rmse, "standard_deviation": "NaN"},
            "MAPE": {"value": mape, "standard_deviation": "NaN"},
            "MASE": {"value": mase, "standard_deviation": "NaN"},
            "WAPE": {"value": wape, "standard_deviation": "NaN"},
        },
    }

    # Output directory and file
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    evaluation_path = os.path.join(output_dir, "evaluation_metrics.json")
    
    # Write the report to a JSON file
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
