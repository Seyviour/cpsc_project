import os
import pandas as pd
import argparse

def merge_pkl_files(folder_path, output_file):
    # List all files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
    
    # Initialize an empty list to store dataframes
    dataframes = []
    
    # Load each .pkl file and append the dataframe to the list
    for file in files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_pickle(file_path)
        dataframes.append(df)
    
    # Merge all dataframes
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    # Save the merged dataframe to a new .pkl file
    merged_df.to_pickle(output_file)
    print(f"Merged dataframe saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge .pkl files in a folder into a single .pkl file.")
    parser.add_argument("input_folder", help="Path to the folder containing .pkl files.")
    parser.add_argument("output_file", help="Path to the output .pkl file (including filename).")
    args = parser.parse_args()
    
    merge_pkl_files(args.input_folder, args.output_file)