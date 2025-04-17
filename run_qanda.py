import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the  data to generate QANDA pairs
pickle_file_path = "val.withextracts.pickle"
val_data = pd.read_pickle(pickle_file_path)

def generate_messages(val_data):
    messages = []
    for _, row in val_data.iterrows():
        claim = row["claim"] 
        relevant_docs = row["relevant_docs"] 
        for doc in relevant_docs:
            message = [
                {
                    "role": "system",
                    "content": "You are a QA generation assistant. Given a claim and its supporting document, generate a relevant question-answer pair.",
                },
                {
                    "role": "user",
                    "content": f"[QANDA] <claim>\n{claim}\n\n<document>\n{doc}",
                },
            ]
            messages.append({"row": row, "message": message, "document": doc})
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

def process_in_batches(messages, model, tokenizer, device, batch_size=300, max_new_tokens=200):
    results = []
    for i in range(0, len(messages), batch_size):
        batch = messages[i:i + batch_size]
        input_texts = [convert_to_chatml(item["message"])["text"] for item in batch]
        input_ids = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
        
        # Generate responses
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
            row = item["row"] 
            document = item["document"] 
            response = tokenizer.decode(
                output_ids[j][input_ids.shape[1]:], skip_special_tokens=True
            )
            qanda = response.replace("assistant", "").strip()
            result = row.to_dict()  
            result["document"] = document 
            result["qanda"] = qanda 
    return results

# Load model and tokenizer
model_path = "jpangas/autotrain-llama-cpsc"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", torch_dtype="auto"
).eval()

device = torch.device("cuda")
model.to(device)

messages = generate_messages(val_data)

results = process_in_batches(messages, model, tokenizer, device)

qanda_df = pd.DataFrame(results)
csv_path = "qanda_results.csv"
qanda_df.to_csv(csv_path, index=False)
print(f"QANDA pairs saved to {csv_path}")

