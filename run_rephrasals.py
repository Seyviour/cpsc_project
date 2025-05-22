# This script is designed to generate multiple rephrased versions of input claims.
# It utilizes a pre-trained causal language model (LLM) from the Hugging Face Transformers library
# to perform the rephrasing task.
#
# The overall workflow is:
# 1. Load input data from a pickle file (e.g., "val.withextracts.pickle"). This file is expected
#    to contain at least a "claim" column with the text of the claims to be rephrased.
# 2. Prepare input messages for the LLM:
#    - For each claim, create a prompt.
#    - The prompt instructs the model (via a system message) to act as a rephrasing assistant and
#      generate three diverse rephrasals that preserve meaning and use correct grammar.
#    - The user part of the prompt provides the original claim text.
# 3. Convert these structured messages into the ChatML format, which is suitable for the chosen
#    pre-trained model (e.g., "jpangas/autotrain-llama-cpsc").
# 4. Load the pre-trained LLM and its tokenizer.
# 5. Process the formatted messages in batches:
#    - Tokenize the ChatML strings.
#    - Feed the tokenized input to the model to generate the rephrased claims.
#    - Decode the model's output to get the textual rephrasals.
# 6. Store the results. Each result will link back to the original claim's data and include
#    the string of generated rephrasals.
# 7. Save these results into a CSV file (`rephrasals_allresults.csv`).

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# --- Configuration ---
# Path to the input pickle file. Expected to contain a DataFrame with a "claim" column.
pickle_file_path = "val.withextracts.pickle" 
# Load the data from the pickle file.
val_data = pd.read_pickle(pickle_file_path)
print(f"Loaded data from {pickle_file_path}. Number of claims to rephrase: {len(val_data)}")

# --- Function Definitions ---

def generate_messages(input_data_df):
    """
    Generates a list of structured messages for the LLM, one for each claim to be rephrased.

    Args:
        input_data_df (pd.DataFrame): DataFrame where each row is expected to have a "claim" column.

    Returns:
        list[dict]: A list of dictionaries. Each dictionary contains:
            - "row": The original row data (Pandas Series) from the input DataFrame.
            - "message": The structured message (list of chat turns) for the LLM, asking for rephrasals.
    """
    messages = []
    print("Generating messages for claim rephrasing...")
    # Iterate through each row (claim) in the DataFrame.
    for index, row in input_data_df.iterrows():
        claim_text = row["claim"] 
        
        # Structure the message for the LLM.
        # System prompt defines the LLM's role and task (generating three diverse rephrasals).
        # User prompt provides the specific claim to be rephrased.
        message_content = [
            {
                "role": "system",
                "content": "You are a rephrasing assistant. Rewrite the given claim into three diverse rephrasals that preserve the meaning without adding new info. Use correct grammar.",
            },
            {
                "role": "user",
                "content": f"[Rewrite] <claim>\n{claim_text}", # Tagging the operation for clarity.
            },
        ]
        # Store the original row data and the message for later reference.
        messages.append({"row": row, "message": message_content}) 
    print(f"Generated {len(messages)} messages for rephrasing.")
    return messages

def convert_to_chatml(chat_data):
    """
    Converts a list of chat turns into the ChatML format string.
    (Identical to the function in other scripts like run_qanda.py, as it serves the same purpose).
    """
    role_tokens = {
        "system": "<|start_header_id|>system<|end_header_id|>",
        "user": "<|start_header_id|>user<|end_header_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>", # For model's response.
    }

    chatml_string = "<|begin_of_text|>" # Start of text token for ChatML.
    for turn in chat_data:
        role = turn["role"]
        content = turn["content"].strip() # Remove leading/trailing whitespace.
        header = role_tokens[role]
        # Append the formatted turn to the ChatML string.
        chatml_string += f"{header}\n\n{content}<|eot_id|>" # <|eot_id|> is the end of turn token.

    return {"text": chatml_string}

def process_in_batches(messages_to_process, model, tokenizer, device, batch_size=200, max_new_tokens=250):
    """
    Processes messages in batches to generate rephrased claims using the language model.

    Args:
        messages_to_process (list[dict]): List of message objects from `generate_messages`.
        model (transformers.PreTrainedModel): The loaded language model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        device (torch.device): The device to run the model on ("cuda" or "cpu").
        batch_size (int): Number of messages (claims) to process in each batch.
        max_new_tokens (int): Maximum number of tokens the model should generate for the rephrasals.
                              This should be enough to accommodate three rephrased versions.

    Returns:
        list[dict]: A list of dictionaries. Each dictionary is an augmented version of the
                    original row data, now including the "rephrasals" string generated by the model.
    """
    results = []
    print(f"Processing {len(messages_to_process)} messages in batches of {batch_size} for rephrasing...")
    for i in range(0, len(messages_to_process), batch_size):
        current_batch = messages_to_process[i:i + batch_size]
        # Convert messages in the current batch to ChatML text format.
        batch_input_texts = [convert_to_chatml(item["message"])["text"] for item in current_batch]
        
        # Tokenize the batch of ChatML strings.
        input_ids = tokenizer(
            batch_input_texts, return_tensors="pt", padding=True, truncation=True
        ).input_ids.to(device) # Move tokenized inputs to the specified device.
        
        # Generate responses (rephrased claims) from the model.
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,      # Max tokens for the generated rephrasals.
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,                     # Use sampling.
            top_p=0.95,                         # Nucleus sampling parameter.
            temperature=0.2,                    # Controls randomness; lower is more deterministic.
        )
        
        # Decode the generated tokens and process the results for this batch.
        for j, item_in_batch in enumerate(current_batch):
            original_row_data = item_in_batch["row"] # This is a Pandas Series.
            
            # Decode only the newly generated part of the output.
            generated_text = tokenizer.decode(
                output_ids[j][input_ids.shape[1]:], skip_special_tokens=True
            )
            # Clean up the generated text (e.g., remove "assistant" prefix if model adds it).
            generated_rephrasals_string = generated_text.replace("assistant", "").strip()
            
            # Convert the original row (Pandas Series) to a dictionary to add the new "rephrasals" field.
            result_dict = original_row_data.to_dict() 
            result_dict["rephrasals"] = generated_rephrasals_string # Add the generated rephrasals string.
            results.append(result_dict)
        print(f"Processed batch {i//batch_size + 1}/{(len(messages_to_process) + batch_size - 1)//batch_size}")
    return results

# --- Main Execution ---

# Load the pre-trained language model and its tokenizer.
# The model "jpangas/autotrain-llama-cpsc" is used here as well.
model_path = "jpangas/autotrain-llama-cpsc"
print(f"Loading tokenizer from: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path)
print(f"Loading model from: {model_path}")
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", torch_dtype="auto", # device_map and dtype for efficiency.
).eval() # Set model to evaluation mode.

# Determine and set the device for PyTorch operations.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# If device_map="auto" didn't handle it or for single-GPU, explicitly move model.
if "auto" not in model.hf_device_map:
    model.to(device)
print(f"Model loaded on device: {model.device}")

# Generate the list of messages (prompts) to be processed by the LLM for rephrasing.
all_messages_for_rephrasing = generate_messages(val_data)

# Process these messages in batches to generate the rephrased claims.
rephrasing_results = process_in_batches(all_messages_for_rephrasing, model, tokenizer, device)

# Convert the list of result dictionaries into a Pandas DataFrame.
# Each row in this DataFrame will contain the original claim information and the string of rephrasals.
rephrasals_df = pd.DataFrame(rephrasing_results)

# Define the path for the output CSV file.
output_csv_path = "rephrasals_allresults.csv"
# Save the DataFrame to CSV. index=False avoids writing the DataFrame index.
rephrasals_df.to_csv(output_csv_path, index=False)

print(f"\nRephrased claims saved to {output_csv_path}")
print(f"Total claims processed for rephrasing: {len(rephrasals_df)}")
print("Sample of generated rephrasals data:")
print(rephrasals_df.head())
