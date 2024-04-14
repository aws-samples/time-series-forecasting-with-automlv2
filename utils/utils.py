import boto3
import os
import pandas as pd


def copy_download_training_data(source_bucket, source_key, destination_bucket, destination_prefix, destination_suffix, download=False):
    
    s3 = boto3.resource('s3')
    copy_source = {
        'Bucket': source_bucket,
        'Key': source_key
    }
    
    s3.meta.client.copy(copy_source, destination_bucket, destination_prefix+destination_suffix)
    
    # Ensure the local directory exists
    local_directory = 'data'
    if not os.path.exists(local_directory):
        os.makedirs(local_directory)
        
    if download: 
        # Initialize a boto3 S3 client
        s3_client = boto3.client('s3')
        object_key = 'automlv2-time-series-data/full-data/synthetic-food-demand.csv'

        # Specify the local file path to save the downloaded file
        local_file_path = os.path.join(local_directory, 'synthetic-food-demand.csv')

        # Download the file
        s3_client.download_file(destination_bucket, object_key, local_file_path)

        print(f"File downloaded successfully to {local_file_path}")
        
        return local_file_path
    else:
        s3_file_path = destination_bucket + "/" + destination_prefix + destination_suffix
        print(f"Data copied to S3 location: {s3_file_path}")
        
        return s3_file_path 
    

def split_train_test(input_data):
    # Ensure the 'timestamp' column is in the correct datetime format (if not already)
    input_data['timestamp'] = pd.to_datetime(input_data['timestamp'])

    # Sort the DataFrame
    input_data_sorted = input_data.sort_values(by=['product_code', 'location_code', 'timestamp'])

    # Lists to collect DataFrames for training and testing
    train_dfs = []
    test_dfs = []

    # Split the data
    for (_, group) in input_data_sorted.groupby(['product_code', 'location_code']):
        test_initial = group.tail(8)  # Initial test set with the last 8 timestamps
        test_rows = test_initial.iloc[:-4]  # Final test set, dropping the last 4 timestamps
        train_rows = group.iloc[:-8]  # The rest for the training set

        # Append groups to the respective lists
        train_dfs.append(train_rows)
        test_dfs.append(test_rows)

    # Concatenate all the DataFrames in the lists to form the final training and test sets
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    # Save the split datasets to CSV files
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)