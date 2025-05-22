# This script serves as a utility to merge multiple Pandas DataFrame pickle files ('.pkl')
# located within a specified input folder into a single, consolidated pickle file.
# This is often useful when data processing tasks are distributed and produce partial results
# (e.g., one .pkl file per worker or per batch), and these partial results need to be
# combined into a final dataset for further analysis or use.
#
# The script uses command-line arguments to specify the input folder and the output file path.

import os # Used for interacting with the file system, like listing files in a directory.
import pandas as pd # The primary library used for data manipulation, specifically for DataFrames and reading/writing pickle files.
import argparse # Used for parsing command-line arguments, making the script flexible and usable from the terminal.

def merge_pkl_files(folder_path, output_file_path):
    """
    Merges all .pkl files found in the specified folder_path into a single .pkl file.

    Args:
        folder_path (str): The path to the folder containing the .pkl files to be merged.
        output_file_path (str): The path (including filename) where the merged .pkl file will be saved.
    """
    # List all files in the given folder_path that end with the '.pkl' extension.
    # This ensures that only Pandas pickle files are considered for merging.
    try:
        files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
    except FileNotFoundError:
        print(f"Error: Input folder '{folder_path}' not found.")
        return
    except Exception as e:
        print(f"Error listing files in '{folder_path}': {e}")
        return

    if not files:
        print(f"No .pkl files found in '{folder_path}'. Nothing to merge.")
        return
        
    print(f"Found {len(files)} .pkl files to merge in '{folder_path}'.")
    
    # Initialize an empty list to store the DataFrames read from each .pkl file.
    dataframes_list = []
    
    # Loop through each identified .pkl file.
    for file_name in files:
        file_full_path = os.path.join(folder_path, file_name)
        try:
            # Read the .pkl file into a Pandas DataFrame.
            df = pd.read_pickle(file_full_path)
            dataframes_list.append(df)
            print(f"Successfully loaded '{file_name}'.")
        except Exception as e:
            # If a file cannot be read (e.g., corrupted or not a valid Pandas pickle),
            # print an error and skip that file.
            print(f"Error reading '{file_full_path}': {e}. Skipping this file.")
            continue # Skip to the next file
    
    # If no DataFrames were successfully loaded (e.g., all files were problematic or folder was empty initially).
    if not dataframes_list:
        print("No valid DataFrames were loaded. Output file will not be created.")
        return

    # Concatenate (merge) all DataFrames in the list into a single DataFrame.
    # `ignore_index=True` re-indexes the resulting DataFrame from 0 to n-1, which is
    # useful if the individual DataFrames had overlapping or non-sequential indices.
    print("Merging DataFrames...")
    merged_df = pd.concat(dataframes_list, ignore_index=True)
    
    # Save the newly merged DataFrame to the specified output .pkl file.
    try:
        merged_df.to_pickle(output_file_path)
        print(f"Successfully merged {len(dataframes_list)} DataFrame(s).")
        print(f"Merged DataFrame saved to '{output_file_path}'. Shape: {merged_df.shape}")
    except Exception as e:
        print(f"Error saving merged DataFrame to '{output_file_path}': {e}")

# This block executes if the script is run directly from the command line.
if __name__ == "__main__":
    # Initialize an ArgumentParser object to handle command-line arguments.
    # The description provides help text when the script is run with -h or --help.
    parser = argparse.ArgumentParser(description="Merge multiple .pkl (Pandas DataFrame pickle) files from a specified folder into a single .pkl file.")
    
    # Define the command-line arguments the script expects:
    # 1. "input_folder": A required positional argument for the path to the folder containing .pkl files.
    parser.add_argument("input_folder", help="Path to the folder containing .pkl files to be merged.")
    # 2. "output_file": A required positional argument for the path where the merged .pkl file will be saved.
    parser.add_argument("output_file", help="Path to the output .pkl file (including the filename, e.g., merged_data.pkl).")
    
    # Parse the arguments provided by the user when running the script.
    args = parser.parse_args()
    
    # Call the main function `merge_pkl_files` with the parsed arguments.
    # This initiates the merging process using the user-specified input folder and output file path.
    merge_pkl_files(args.input_folder, args.output_file)