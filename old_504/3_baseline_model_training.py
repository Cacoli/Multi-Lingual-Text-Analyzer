# 3_baseline_model_training.py (Corrected)
#
# --- GOAL ---
# This script builds the multi-task model architecture, loads the processed data,
# and trains a functional baseline version of the model on a GPU.

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel 
from torch.optim import AdamW 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os

# --- Configuration ---
class Config:
    INPUT_FILENAME = "final_processed_dataset.csv"
    MODEL_NAME = "xlm-roberta-base"
    OUTPUT_MODEL_PATH = "baseline_model.pth"
    MAX_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5

# --- 1. Custom Dataset Class ---
class TweetDataset(Dataset):
    def __init__(self, texts, hate_labels, sarcasm_labels, emotion_labels, tokenizer, max_len):
        self.texts = texts
        self.hate_labels = hate_labels
        self.sarcasm_labels = sarcasm_labels
        self.emotion_labels = emotion_labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'hate_labels': torch.tensor(self.hate_labels[item], dtype=torch.long),
            'sarcasm_labels': torch.tensor(self.sarcasm_labels[item], dtype=torch.long),
            'emotion_labels': torch.tensor(self.emotion_labels[item], dtype=torch.long)
        }

# --- 2. Model Architecture ---
class MultiTaskModel(nn.Module):
    def __init__(self, base_model, num_hate, num_sarcasm, num_emotion):
        super(MultiTaskModel, self).__init__()
        self.base = base_model
        hidden_size = self.base.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.hate_classifier = nn.Linear(hidden_size, num_hate)
        self.sarcasm_classifier = nn.Linear(hidden_size, num_sarcasm)
        self.emotion_classifier = nn.Linear(hidden_size, num_emotion)

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        hate_logits = self.hate_classifier(pooled_output)
        sarcasm_logits = self.sarcasm_classifier(pooled_output)
        emotion_logits = self.emotion_classifier(pooled_output)
        return hate_logits, sarcasm_logits, emotion_logits

# --- 3. Main Training Function ---
def train_model():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Training on CPU will be extremely slow.")
    print(f"Using device: {device}")

    df = pd.read_csv(config.INPUT_FILENAME).dropna()
    hate_encoder = LabelEncoder()
    sarcasm_encoder = LabelEncoder()
    emotion_encoder = LabelEncoder()
    df['hate_encoded'] = hate_encoder.fit_transform(df['hate_speech'])
    df['sarcasm_encoded'] = sarcasm_encoder.fit_transform(df['sarcasm'])
    df['emotion_encoded'] = emotion_encoder.fit_transform(df['emotion'])
    
    df_train, df_val = train_test_split(df, test_size=0.1, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    train_dataset = TweetDataset(
        texts=df_train.text.to_numpy(),
        hate_labels=df_train.hate_encoded.to_numpy(),
        sarcasm_labels=df_train.sarcasm_encoded.to_numpy(),
        emotion_labels=df_train.emotion_encoded.to_numpy(),
        tokenizer=tokenizer,
        max_len=config.MAX_LEN
    )
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    base_model = AutoModel.from_pretrained(config.MODEL_NAME)
    model = MultiTaskModel(
        base_model=base_model,
        num_hate=len(hate_encoder.classes_),
        num_sarcasm=len(sarcasm_encoder.classes_),
        num_emotion=len(emotion_encoder.classes_)
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{config.EPOCHS} ---")
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            hate_labels = batch['hate_labels'].to(device)
            sarcasm_labels = batch['sarcasm_labels'].to(device)
            emotion_labels = batch['emotion_labels'].to(device)

            optimizer.zero_grad()
            hate_logits, sarcasm_logits, emotion_logits = model(input_ids, attention_mask)
            
            loss_hate = loss_fn(hate_logits, hate_labels)
            loss_sarcasm = loss_fn(sarcasm_logits, sarcasm_labels)
            loss_emotion = loss_fn(emotion_logits, emotion_labels)
            
            loss = loss_hate + loss_sarcasm + loss_emotion
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch + 1} | Average Training Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), config.OUTPUT_MODEL_PATH)

if __name__ == "__main__":
    print("\n--- STEP 3A: BASELINE MODEL TRAINING STARTING ---")
    train_model()
    print("\n--- SCRIPT COMPLETE ---")
    if os.path.exists(Config.OUTPUT_MODEL_PATH):
        print(f"Baseline model training is complete.")
        print(f"Model saved to: {os.path.abspath(Config.OUTPUT_MODEL_PATH)}")
        print("\n--- NEXT STEP ---")
        print("We now have a functional model! The next step is to enhance and fine-tune it.")
    else:
        print("Script finished, but no model file was created.")