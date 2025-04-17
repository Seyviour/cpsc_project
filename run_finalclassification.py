import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

qanda_csv_path = "qanda_results.csv"  
qanda_data = pd.read_csv(qanda_csv_path)

def prepare_classification_messages(qanda_data):
    messages = []
    # Group QANDA pairs by claim_id
    grouped = qanda_data.groupby("claim_id")
    print(f"Number of unique claim_ids: {len(grouped)}")
    for claim_id, group in grouped:
        claim = group["claim"].iloc[0]  
        label = group["label"].iloc[0] 
        qandas = "\n".join(group["qanda"].dropna())  
        
        # Create the classification message
        message = [
            {
                "role": "system",
                "content": "You are a classification assistant. Given a claim and QA pairs based on retrieved evidence, classify the claim as one of the following: Supported, Refuted, Conflicting Evidence/Cherrypicking, Not Enough Evidence.",
            },
            {
                "role": "user",
                "content": f"[Classify] <claim>\n{claim}\n\n{qandas}\n\nlabel: ",
            },
        ]
        messages.append({"claim_id": claim_id, "claim": claim, "message": message, "label": label})
    return messages

def convert_to_chatml(chat_data):
    role_tokens = {
        "system": "<|start_header_id|>system<|end_header_id|>",
        "user": "<|start_header_id|>user<|end_header_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>",
    }

    chatml = "<|begin_of_text|>"
    for turn in chat_data:
        role = turn["role"]
        content = turn["content"].strip()
        header = role_tokens[role]
        chatml += f"{header}\n\n{content}<|eot_id|>"

    return {"text": chatml}

def process_in_batches(messages, model, tokenizer, device, batch_size=100, max_new_tokens=150):
    results = []
    for i in range(0, len(messages), batch_size):
        batch = messages[i:i + batch_size]
        input_texts = [convert_to_chatml(item["message"])["text"] for item in batch]
        input_ids = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
        
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.95,
            temperature=0.2,
        )
        
        # Decode and process results
        for j, item in enumerate(batch):
            claim_id = item["claim_id"]
            claim = item["claim"]
            label = item["label"]
            response = tokenizer.decode(
                output_ids[j][input_ids.shape[1]:], skip_special_tokens=True
            )
            classification = response.replace("assistant", "").strip()
            results.append({
                "claim_id": claim_id,
                "claim": claim,
                "classification": classification,
                "label": label
            })
    return results

# Load model and tokenizer
model_path = "jpangas/autotrain-llama-cpsc"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", torch_dtype="auto",
).eval()

device = torch.device("cuda")
model.to(device)

messages = prepare_classification_messages(qanda_data)

results = process_in_batches(messages, model, tokenizer, device)

classification_df = pd.DataFrame(results)
classification_csv_path = "classification_results.csv"
classification_df.to_csv(classification_csv_path, index=False)
print(f"Classification results saved to {classification_csv_path}")