import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics(data, true_col, pred_col):
    """Calculate MAE, RMSE, R², MAPE, F1, precision, recall, and confusion matrix."""
    mae = mean_absolute_error(data[true_col], data[pred_col])
    rmse = np.sqrt(mean_squared_error(data[true_col], data[pred_col]))
    r2 = r2_score(data[true_col], data[pred_col])
    mape = np.mean(np.abs((data[true_col] - data[pred_col]) / data[true_col])) * 100 if np.any(data[true_col] != 0) else float('inf')
    
    # Calculate F1, Precision, and Recall for binary classification
    precision = precision_score(data[true_col] > 0, data[pred_col] > 0, zero_division=0)
    recall = recall_score(data[true_col] > 0, data[pred_col] > 0)
    f1 = f1_score(data[true_col] > 0, data[pred_col] > 0, zero_division=0)
    
    conf_mat = confusion_matrix(data[true_col] > 0, data[pred_col] > 0)
    
    return mae, rmse, r2, mape, precision, recall, f1, conf_mat

import matplotlib.pyplot as plt

def plot_predictions_vs_actuals(data, true_col, pred_col):
    plt.figure(figsize=(10, 6))
    plt.scatter(data[true_col], data[pred_col], alpha=0.5)
    plt.plot([data[true_col].min(), data[true_col].max()], [data[true_col].min(), data[true_col].max()], 'r--')  # Ideal line
    plt.title(f'Predicted vs. Actual Counts for {pred_col}')
    plt.xlabel('Actual Count')
    plt.ylabel('Predicted Count')
    plt.grid(True)
    plt.savefig("output/pred-vs-act-"+pred_col+".png", format='png', dpi=300, bbox_inches='tight')
    plt.close()

# Assuming df is your DataFrame loaded from the CSV



def main():
    # Load the data
    df = pd.read_csv("updated_val_predictions.csv")
    
    # List of prediction columns, dynamically extracted
    prediction_columns = [col for col in df.columns if 'pred' in col]
    
    # Calculate metrics for each prediction model
    metrics = {}
    for pred_col in prediction_columns:
        mae, rmse, r2, mape, precision, recall, f1, conf_mat = calculate_metrics(df, 'person_count', pred_col)
        metrics[pred_col] = {
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'MAPE': mape,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Confusion Matrix': conf_mat
        }
        # Plot confusion matrix
        plot_predictions_vs_actuals(df, 'person_count', pred_col)

        
        # Print metrics
        print(f"Metrics for {pred_col}:")
        print(f"MAE: {mae}, RMSE: {rmse}, R²: {r2}, MAPE: {mape}, Precision: {precision}, Recall: {recall}, F1: {f1}\n")

if __name__ == "__main__":
    main()

