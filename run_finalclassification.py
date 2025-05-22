# This script classifies claims based on Question and Answer (QANDA) pairs generated from retrieved evidence.
# It utilizes a pre-trained causal language model (LLM) from the Hugging Face Transformers library.
#
# The overall workflow is:
# 1. Load QANDA pairs from a CSV file (`qanda_results.csv`). Each row should contain a claim,
#    its associated QANDA pair (derived from evidence), a claim_id, and an original label.
# 2. Prepare input messages for the LLM:
#    - For each unique claim, aggregate all its QANDA pairs.
#    - Format these into a structured message that includes a system prompt (instructing the model
#      on its classification task) and a user prompt (containing the claim and the aggregated QANDA pairs).
# 3. Convert these structured messages into the ChatML format, which is a specific text-based
#    representation required by the chosen pre-trained model.
# 4. Load the pre-trained LLM (e.g., "jpangas/autotrain-llama-cpsc") and its tokenizer.
# 5. Process the formatted messages in batches:
#    - Tokenize the ChatML strings.
#    - Feed the tokenized input to the model to generate classifications.
#    - Decode the model's output to get the textual classification.
# 6. Store the results (claim_id, claim, model's classification, original label) in a Pandas DataFrame.
# 7. Save the DataFrame to a new CSV file (`classification_results.csv`).

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# --- Configuration ---
# Path to the input CSV file containing QANDA pairs, claims, claim_ids, and labels.
# Expected columns: "claim_id", "claim", "qanda", "label".
qanda_csv_path = "qanda_results.csv"  
# Load the QANDA data from the CSV file into a Pandas DataFrame.
qanda_data = pd.read_csv(qanda_csv_path)

# --- Function Definitions ---

def prepare_classification_messages(qanda_data):
    """
    Prepares structured messages for claim classification based on QANDA pairs.

    Each message is a list of dictionaries, following a chat-like structure (system, user turns),
    designed to prompt the language model for classification.

    Args:
        qanda_data (pd.DataFrame): DataFrame containing columns "claim_id", "claim", 
                                   "qanda", and "label".

    Returns:
        list[dict]: A list of dictionaries. Each dictionary contains:
            - "claim_id": The ID of the claim.
            - "claim": The text of the claim.
            - "message": A list of chat turns (system prompt and user content with claim + QANDAs).
            - "label": The original label of the claim.
    """
    messages = []
    # Group QANDA pairs by "claim_id" to process each claim and its associated QANDAs together.
    grouped = qanda_data.groupby("claim_id")
    print(f"Number of unique claim_ids to process: {len(grouped)}")

    for claim_id, group in grouped:
        claim = group["claim"].iloc[0]  # Get the claim text (should be the same for all rows in a group).
        label = group["label"].iloc[0] # Get the original label.
        # Concatenate all non-null QANDA pairs for the current claim into a single string.
        qandas = "\n".join(group["qanda"].dropna())  
        
        # Create the structured message for the language model.
        # This includes a system prompt defining the task and expected output categories,
        # and a user prompt containing the specific claim and its QANDA evidence.
        message_content = [
            {
                "role": "system",
                "content": "You are a classification assistant. Given a claim and QA pairs based on retrieved evidence, classify the claim as one of the following: Supported, Refuted, Conflicting Evidence/Cherrypicking, Not Enough Evidence.",
            },
            {
                "role": "user",
                # The user content includes the claim and the aggregated QANDA pairs.
                # The "label: " part at the end prompts the model to fill in the classification.
                "content": f"[Classify] <claim>\n{claim}\n\n{qandas}\n\nlabel: ",
            },
        ]
        messages.append({"claim_id": claim_id, "claim": claim, "message": message_content, "label": label})
    return messages

def convert_to_chatml(chat_data):
    """
    Converts a list of chat turns (dictionaries with "role" and "content") into the ChatML format.

    ChatML is a specific text format that uses special tokens to delineate turns and roles
    (e.g., system, user, assistant). This format is expected by some language models.

    Args:
        chat_data (list[dict]): A list of chat turns, where each turn is a dictionary
                                with "role" (e.g., "system", "user") and "content".

    Returns:
        dict: A dictionary with a single key "text", whose value is the ChatML formatted string.
    """
    # Define the specific header tokens for each role in ChatML.
    role_tokens = {
        "system": "<|start_header_id|>system<|end_header_id|>",
        "user": "<|start_header_id|>user<|end_header_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>", # Though assistant turn is not in input here, it's part of ChatML spec.
    }

    chatml_string = "<|begin_of_text|>" # Start of text token for ChatML.
    for turn in chat_data:
        role = turn["role"]
        content = turn["content"].strip() # Remove leading/trailing whitespace from content.
        header = role_tokens[role]
        # Append the formatted turn to the ChatML string.
        chatml_string += f"{header}\n\n{content}<|eot_id|>" # <|eot_id|> is the end of turn token.

    return {"text": chatml_string}

def process_in_batches(messages_data, model, tokenizer, device, batch_size=100, max_new_tokens=150):
    """
    Processes messages in batches to generate classifications using the language model.

    Args:
        messages_data (list[dict]): List of message objects from `prepare_classification_messages`.
        model (transformers.PreTrainedModel): The loaded language model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        device (torch.device): The device to run the model on (e.g., "cuda" or "cpu").
        batch_size (int): Number of messages to process in each batch.
        max_new_tokens (int): Maximum number of tokens the model should generate for the classification.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains:
            - "claim_id": ID of the claim.
            - "claim": Text of the claim.
            - "classification": The classification generated by the model.
            - "label": The original label of the claim.
    """
    results = []
    print(f"Processing {len(messages_data)} messages in batches of {batch_size}...")
    for i in range(0, len(messages_data), batch_size):
        batch_items = messages_data[i:i + batch_size]
        # Convert each message in the batch to ChatML format.
        input_texts_chatml = [convert_to_chatml(item["message"])["text"] for item in batch_items]
        
        # Tokenize the batch of ChatML strings.
        # `padding=True` pads shorter sequences to the length of the longest in the batch.
        # `truncation=True` truncates sequences longer than the model's max input length.
        # `return_tensors="pt"` returns PyTorch tensors.
        input_ids = tokenizer(
            input_texts_chatml, return_tensors="pt", padding=True, truncation=True
        ).input_ids.to(device) # Move tokenized inputs to the specified device.
        
        # Generate responses from the model.
        # `pad_token_id=tokenizer.eos_token_id` is important for proper generation.
        # `do_sample=True`, `top_p`, `temperature` control the sampling strategy for generation.
        # Adjust these parameters to tune creativity vs. determinism.
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,      # Max tokens for the classification label itself.
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,                     # Use sampling for generation.
            top_p=0.95,                         # Nucleus sampling parameter.
            temperature=0.2,                    # Controls randomness; lower is more deterministic.
        )
        
        # Decode the generated token IDs back to text and process results.
        for j, current_item in enumerate(batch_items):
            claim_id = current_item["claim_id"]
            claim_text = current_item["claim"]
            original_label = current_item["label"]
            
            # Decode only the newly generated tokens (response part).
            # `output_ids[j][input_ids.shape[1]:]` slices the output to get only the generated part.
            response_text = tokenizer.decode(
                output_ids[j][input_ids.shape[1]:], skip_special_tokens=True
            )
            # Clean up the response: remove "assistant" prefix if present and strip whitespace.
            # The model might generate "assistant\nClassification Label".
            generated_classification = response_text.replace("assistant", "").strip()
            
            results.append({
                "claim_id": claim_id,
                "claim": claim_text,
                "classification": generated_classification, # This is the model's output.
                "label": original_label # This is the original label from input data, for comparison.
            })
        print(f"Processed batch {i//batch_size + 1}/{(len(messages_data) + batch_size - 1)//batch_size}")
    return results

# --- Main Execution ---

# Load the pre-trained language model and its tokenizer.
# "jpangas/autotrain-llama-cpsc" is a specific model fine-tuned for a similar task (presumably).
# `device_map="auto"` attempts to distribute the model across available hardware (GPUs, CPU) automatically.
# `torch_dtype="auto"` selects the optimal data type (e.g., float16 for faster inference on compatible GPUs).
# `.eval()` sets the model to evaluation mode (disables dropout, etc.).
model_path = "jpangas/autotrain-llama-cpsc" 
print(f"Loading tokenizer from: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path)
print(f"Loading model from: {model_path}")
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", torch_dtype="auto",
).eval()

# Set the device for model operations (prefer CUDA if available).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Explicitly move the model to the device if device_map="auto" didn't handle it or for single-GPU setup.
# If device_map="auto" worked, this might be redundant but usually harmless.
if "auto" not in model.hf_device_map: # Check if model is already device mapped
    model.to(device)
print(f"Model loaded on device: {model.device}")


# Prepare the messages for classification.
print("Preparing classification messages...")
messages_for_classification = prepare_classification_messages(qanda_data)

# Process the messages in batches to get classifications.
print("Starting classification processing...")
classification_results = process_in_batches(messages_for_classification, model, tokenizer, device)

# Convert the list of result dictionaries into a Pandas DataFrame.
classification_df = pd.DataFrame(classification_results)

# Define the path for the output CSV file.
classification_csv_path = "classification_results.csv"
# Save the DataFrame to a CSV file. `index=False` prevents writing DataFrame index as a column.
classification_df.to_csv(classification_csv_path, index=False)

print(f"\nClassification results saved to {classification_csv_path}")
print(f"Number of claims classified: {len(classification_df)}")
print("Sample results:")
print(classification_df.head())