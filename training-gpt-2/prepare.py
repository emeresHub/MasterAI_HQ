import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import GPT2Tokenizer

# Configuration
data_path = 'data/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv'  # Original CSV file
output_dir = 'data/Bitext-customer-support-llm-chatbot-training-dataset'
tokenizer_output_dir = '/content/training-gpt-2/training-gpt-2/out'  # Save tokenizer here for reuse
train_split = 0.9  # 90% training, 10% validation

# Ensure output directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(tokenizer_output_dir, exist_ok=True)

# Load dataset
df = pd.read_csv(data_path)

# Combine instruction and response into a single training sample
def format_row(row):
    return f"### Instruction:\n{row['instruction']}\n\n### Response:\n{row['response']}\n"

# Prepare full dataset text
all_texts = df.apply(format_row, axis=1).tolist()

# Shuffle and split
np.random.seed(42)
np.random.shuffle(all_texts)
split_idx = int(train_split * len(all_texts))
train_texts = all_texts[:split_idx]
val_texts = all_texts[split_idx:]

# Load tokenizer (GPT-2 style)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add pad token if not present

# Save tokenizer for later use (e.g. during fine-tuning)
tokenizer.save_pretrained(tokenizer_output_dir)

# Tokenize and encode
def encode(texts):
    ids = []
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        ids.extend(tokens + [tokenizer.eos_token_id])  # append EOS
    return np.array(ids, dtype=np.uint16)

train_ids = encode(train_texts)
val_ids = encode(val_texts)

# Save token ids to .bin
train_bin_path = os.path.join(output_dir, 'train.bin')
val_bin_path = os.path.join(output_dir, 'val.bin')
train_ids.tofile(train_bin_path)
val_ids.tofile(val_bin_path)

# Save metadata
meta = {
    'vocab_size': tokenizer.vocab_size,
    'eos_token_id': tokenizer.eos_token_id,
    'pad_token_id': tokenizer.pad_token_id,
}
with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("✅ Preparation complete. Files and tokenizer saved to:")
print(f"👉 Encoded data: {output_dir}")
print(f"👉 Tokenizer: {tokenizer_output_dir}")
