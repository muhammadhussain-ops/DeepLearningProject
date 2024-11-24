import pandas as pd
import glob
import time
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
import os
import argparse

def process_parquet_files(path, output_file, window_size=60, num_cores=-1):
    start_time = time.time()  # Track total time

    # Step 1: Load all parquet files matching the pattern
    all_files = glob.glob(path + "cleaned_data_*.parquet")
    if not all_files:
        print(f"No files found in the directory {path}. Exiting.")
        return

    # Step 2: Define the columns to drop (conditionally)
    drop_columns_set_1 = ['State', 'Type', 'Event']
    drop_columns_set_2 = drop_columns_set_1 + ['main_fault']

    # Step 3: Initialize a list to store dataframes
    dfs = []

    # Step 4: Load, preprocess, and concatenate data
    for file in all_files:
        df = pd.read_parquet(file)

        # Drop columns that exist in the dataframe
        if 'main_fault' in df.columns:
            df.drop(columns=[col for col in drop_columns_set_2 if col in df.columns], inplace=True)
        else:
            df.drop(columns=[col for col in drop_columns_set_1 if col in df.columns], inplace=True)

        dfs.append(df)

    # Step 5: Combine all dataframes into one
    combined_df = pd.concat(dfs, axis=0, ignore_index=True)

    # Step 6: Ensure datetime format
    combined_df['Datetime'] = pd.to_datetime(combined_df['Datetime'])

    # Step 7: Identify continuous sequences
    combined_df['time_diff'] = combined_df['Datetime'].diff().dt.total_seconds()
    combined_df['is_continuous'] = (combined_df['time_diff'].between(59, 61)) | (combined_df.index == 0)
    combined_df['sequence_group'] = (~combined_df['is_continuous']).cumsum()

    # Step 8: Filter only sequences of at least x continuous rows
    valid_sequences = combined_df.groupby('sequence_group').filter(lambda x: len(x) >= window_size)

    # Step 9: Create time-windowed and flattened data
    def process_group(group):
        group_windows = []
        for start_idx in range(0, len(group) - window_size + 1):
            window = group.iloc[start_idx:start_idx + window_size]
            flattened_window = window.drop(
                ['Datetime', 'time_diff', 'is_continuous', 'sequence_group'], axis=1
            ).values.flatten()
            group_windows.append(flattened_window)
        return group_windows

    groups = [group for _, group in valid_sequences.groupby('sequence_group')]

    print(f"Processing {len(groups)} groups using {num_cores} cores...")
    time_windows = Parallel(n_jobs=num_cores)(delayed(process_group)(group) for group in groups)
    time_windows = [window for group_windows in time_windows for window in group_windows]

    # Step 10: Normalize each window
    scaler = MinMaxScaler()
    time_windows_normalized = [scaler.fit_transform(window.reshape(-1, 1)).flatten() for window in time_windows]

    elapsed_time = time.time() - start_time
    print(f"Step 10 time: {elapsed_time:.2f} seconds")

    # Step 11: Save the prepared data
    prepared_data = pd.DataFrame(time_windows_normalized)
    prepared_data.to_parquet(output_file, index=False)

    elapsed_time = time.time() - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Process cleaned data from parquet files.")
    parser.add_argument("path", type=str, help="Directory path containing the parquet files.")
    parser.add_argument("output_file", type=str, help="Output file for the processed data.")
    parser.add_argument("--window_size", type=int, default=60, help="Window size for processing (default: 60).")
    parser.add_argument("--num_cores", type=int, default=-1, help="Number of cores for parallel processing (default: -1).")
    args = parser.parse_args()

    # Call the processing function with parsed arguments
    process_parquet_files(args.path, args.output_file, args.window_size, args.num_cores)