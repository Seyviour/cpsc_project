# This script generates Question and Answer (QANDA) pairs based on claims and their relevant evidence documents.
# It uses a pre-trained causal language model (LLM) from the Hugging Face Transformers library to generate these pairs.
#
# The overall workflow is:
# 1. Load input data from a pickle file (e.g., "val.withextracts.pickle"). This file is expected
#    to contain claims and a list of relevant document extracts for each claim.
# 2. Prepare input messages for the LLM:
#    - For each claim and *each* of its relevant documents, create a separate prompt.
#    - Each prompt instructs the model (via a system message) to act as a QA generation assistant.
#    - The user part of the prompt provides the claim and the content of one specific supporting document.
# 3. Convert these structured messages into the ChatML format, suitable for the chosen pre-trained model.
# 4. Load the pre-trained LLM (e.g., "jpangas/autotrain-llama-cpsc") and its tokenizer.
# 5. Process the formatted messages in batches:
#    - Tokenize the ChatML strings.
#    - Feed the tokenized input to the model to generate QANDA pairs.
#    - Decode the model's output to get the textual QANDA pair.
# 6. Store the results. Each result will link back to the original claim's data, the specific document
#    used for generation, and the generated QANDA pair.
# 7. Save these results into a CSV file (`qanda_results.csv`).

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# --- Configuration ---
# Path to the input pickle file. This file should contain a Pandas DataFrame or a similar structure
# where each row has at least "claim" and "relevant_docs" (a list of text extracts).
pickle_file_path = "val.withextracts.pickle"
# Load the data.
val_data = pd.read_pickle(pickle_file_path)
print(f"Loaded data from {pickle_file_path}. Number of claims: {len(val_data)}")

# --- Function Definitions ---

def generate_messages(input_data_df):
    """
    Generates a list of structured messages for the LLM, one for each claim-document pair.

    Args:
        input_data_df (pd.DataFrame): DataFrame where each row contains a "claim" and
                                      a list of "relevant_docs".

    Returns:
        list[dict]: A list of dictionaries. Each dictionary contains:
            - "row": The original row data (Pandas Series) from the input DataFrame.
            - "message": The structured message (list of chat turns) for the LLM.
            - "document": The specific document text used for this message.
    """
    messages = []
    print(f"Generating messages for QANDA generation...")
    # Iterate through each row (claim and its associated documents) in the DataFrame.
    for index, row in input_data_df.iterrows():
        claim_text = row["claim"] 
        relevant_document_extracts = row["relevant_docs"] 
        
        # For each relevant document associated with the claim, create a separate message.
        for doc_text in relevant_document_extracts:
            # Structure the message for the LLM.
            # System prompt defines the LLM's role.
            # User prompt provides the claim and one document extract.
            message_content = [
                {
                    "role": "system",
                    "content": "You are a QA generation assistant. Given a claim and its supporting document, generate a relevant question-answer pair.",
                },
                {
                    "role": "user",
                    "content": f"[QANDA] <claim>\n{claim_text}\n\n<document>\n{doc_text}",
                },
            ]
            # Store the original row data, the message, and the specific document for later reference.
            messages.append({"row": row, "message": message_content, "document": doc_text})
    print(f"Generated {len(messages)} messages (one per claim-document pair).")
    return messages

def convert_to_chatml(chat_data):
    """
    Converts a list of chat turns into the ChatML format string.
    (Identical to the function in run_finalclassification.py, as it serves the same purpose).
    """
    role_tokens = {
        "system": "<|start_header_id|>system<|end_header_id|>",
        "user": "<|start_header_id|>user<|end_header_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>", # For model's response.
    }

    chatml_string = "<|begin_of_text|>"
    for turn in chat_data:
        role = turn["role"]
        content = turn["content"].strip()
        header = role_tokens[role]
        chatml_string += f"{header}\n\n{content}<|eot_id|>"

    return {"text": chatml_string}

def process_in_batches(messages_to_process, model, tokenizer, device, batch_size=300, max_new_tokens=200):
    """
    Processes messages in batches to generate QANDA pairs using the language model.

    Args:
        messages_to_process (list[dict]): List of message objects from `generate_messages`.
        model (transformers.PreTrainedModel): The loaded language model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        device (torch.device): The device to run the model on ("cuda" or "cpu").
        batch_size (int): Number of messages to process in each batch.
        max_new_tokens (int): Maximum number of tokens the model should generate for the QANDA pair.

    Returns:
        list[dict]: A list of dictionaries. Each dictionary is an augmented version of the
                    original row data, now including the "document" used and the "qanda" pair generated.
    """
    results = []
    print(f"Processing {len(messages_to_process)} messages in batches of {batch_size} for QANDA generation...")
    for i in range(0, len(messages_to_process), batch_size):
        current_batch = messages_to_process[i:i + batch_size]
        # Convert messages in the current batch to ChatML text format.
        batch_input_texts = [convert_to_chatml(item["message"])["text"] for item in current_batch]
        
        # Tokenize the batch of ChatML strings.
        input_ids = tokenizer(
            batch_input_texts, return_tensors="pt", padding=True, truncation=True
        ).input_ids.to(device)
        
        # Generate responses (QANDA pairs) from the model.
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,      # Max tokens for the generated QANDA pair.
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,                     # Use sampling.
            top_p=0.95,                         # Nucleus sampling.
            temperature=0.2,                    # Control randomness.
        )
        
        # Decode the generated tokens and process the results for this batch.
        for j, item_in_batch in enumerate(current_batch):
            original_row_data = item_in_batch["row"] # This is a Pandas Series.
            document_used = item_in_batch["document"] 
            
            # Decode only the newly generated part of the output.
            generated_text = tokenizer.decode(
                output_ids[j][input_ids.shape[1]:], skip_special_tokens=True
            )
            # Clean up the generated text (e.g., remove "assistant" prefix if model adds it).
            generated_qanda_pair = generated_text.replace("assistant", "").strip()
            
            # Convert the original row (Pandas Series) to a dictionary to add new fields.
            result_dict = original_row_data.to_dict()  
            result_dict["document"] = document_used  # Add the specific document used for this QANDA.
            result_dict["qanda"] = generated_qanda_pair # Add the generated QANDA pair.
            results.append(result_dict)
        print(f"Processed batch {i//batch_size + 1}/{(len(messages_to_process) + batch_size - 1)//batch_size}")
    return results

# --- Main Execution ---

# Load the pre-trained language model and its tokenizer.
# Model "jpangas/autotrain-llama-cpsc" is used, same as in the classification script.
model_path = "jpangas/autotrain-llama-cpsc"
print(f"Loading tokenizer from: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path)
print(f"Loading model from: {model_path}")
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", torch_dtype="auto" # device_map and dtype for efficiency.
).eval() # Set model to evaluation mode.

# Determine and set the device for PyTorch operations.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# If device_map="auto" didn't handle it or for single-GPU, explicitly move model.
if "auto" not in model.hf_device_map:
    model.to(device)
print(f"Model loaded on device: {model.device}")

# Generate the list of messages to be processed by the LLM.
all_messages = generate_messages(val_data)

# Process these messages in batches to generate QANDA pairs.
qanda_generation_results = process_in_batches(all_messages, model, tokenizer, device)

# Convert the list of result dictionaries into a Pandas DataFrame.
# Each row in this DataFrame will contain the original claim info, the specific document used,
# and the QANDA pair generated from that claim-document pair.
qanda_df = pd.DataFrame(qanda_generation_results)

# Define the path for the output CSV file.
output_csv_path = "qanda_results.csv"
# Save the DataFrame to CSV. index=False avoids writing the DataFrame index.
qanda_df.to_csv(output_csv_path, index=False)

print(f"\nQANDA pairs saved to {output_csv_path}")
print(f"Total QANDA pairs generated: {len(qanda_df)}")
print("Sample of generated QANDA data:")
print(qanda_df.head())

