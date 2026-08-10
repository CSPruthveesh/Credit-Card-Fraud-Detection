import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

def build_feast_parquet():
    """
    Reads the raw creditcard.csv, runs the feature engineering pipeline,
    creates event_timestamp and transaction_id fields, and saves it
    to data/processed_transactions.parquet for Feast integration.
    """
    csv_path = "creditcard.csv"
    output_dir = "data"
    output_path = os.path.join(output_dir, "processed_transactions.parquet")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError("creditcard.csv not found in the workspace root directory.")
        
    print("Loading raw transactions dataset...")
    df = pd.read_csv(csv_path)
    
    print("Engineering temporal features...")
    # Chronological sort to compute deltas correctly
    df = df.sort_values(by="Time").reset_index(drop=True)
    
    # Reconstruct user proxy
    df['Card_Proxy'] = df['V1'].round(1).astype(str) + "_" + df['V2'].round(1).astype(str)
    
    # Calculate velocity metrics
    df['Time_Delta'] = df.groupby('Card_Proxy')['Time'].diff().fillna(-1)
    df['Last_5_Tx_Time_Span'] = df.groupby('Card_Proxy')['Time'].diff(periods=5).fillna(-1)
    df['Rolling_Mean_Amount_5'] = df.groupby('Card_Proxy')['Amount'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df.drop('Card_Proxy', axis=1, inplace=True)
    
    # Scale Amount and Time (for Feast offline features representation)
    df['Amount_scaled'] = StandardScaler().fit_transform(df[['Amount']])
    df['Time_scaled'] = StandardScaler().fit_transform(df[['Time']])
    
    # Feast requirements: event_timestamp, created_timestamp, and unique entity key
    # Generate event_timestamp: map Time (seconds since first transaction) to real datetime
    base_time = datetime(2026, 8, 10, 0, 0, 0)
    df['event_timestamp'] = df['Time'].apply(lambda t: base_time + timedelta(seconds=int(t)))
    df['created_timestamp'] = datetime.now()
    
    # Create transaction_id (join key entity)
    df['tx_id'] = df.index.astype(np.int64)
    
    # Create directories if they do not exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Write to Parquet file
    print(f"Saving processed features to {output_path}...")
    # Keep only the columns specified in the Feast schema to prevent schema validation issues
    cols_to_save = ['tx_id', 'event_timestamp', 'created_timestamp', 'Time_Delta', 'Last_5_Tx_Time_Span', 'Rolling_Mean_Amount_5', 'Amount_scaled', 'Time_scaled']
    df[cols_to_save].to_parquet(output_path, index=False)
    print("Feast offline parquet source created successfully!")

if __name__ == "__main__":
    build_feast_parquet()
