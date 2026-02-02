# 4_model_enhancement.py
#
# --- GOAL ---
# This script loads the baseline model and fine-tunes it further using advanced
# techniques like weighted loss to handle data imbalance, producing the final model.

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import numpy as np
import os

# --- Configuration ---
class Config:
    INPUT_FILENAME = "final_processed_dataset.csv"
    BASELINE_MODEL_PATH = "baseline_model.pth"
    FINAL_MODEL_PATH = "final_model.pth"
    MODEL_NAME = "xlm-roberta-base"
    MAX_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 5
    LEARNING_RATE = 5e-6

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

def enhance_model():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: {device}")

    df = pd.read_csv(config.INPUT_FILENAME).dropna()
    hate_encoder = LabelEncoder()
    sarcasm_encoder = LabelEncoder()
    emotion_encoder = LabelEncoder()
    df['hate_encoded'] = hate_encoder.fit_transform(df['hate_speech'])
    df['sarcasm_encoded'] = sarcasm_encoder.fit_transform(df['sarcasm'])
    df['emotion_encoded'] = emotion_encoder.fit_transform(df['emotion'])
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    full_dataset = TweetDataset(
        texts=df.text.to_numpy(),
        hate_labels=df.hate_encoded.to_numpy(),
        sarcasm_labels=df.sarcasm_encoded.to_numpy(),
        emotion_labels=df.emotion_encoded.to_numpy(),
        tokenizer=tokenizer,
        max_len=config.MAX_LEN
    )
    train_loader = DataLoader(full_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    print("Calculating class weights to handle data imbalance...")
    hate_weights = compute_class_weight('balanced', classes=np.unique(df['hate_encoded']), y=df['hate_encoded'])
    sarcasm_weights = compute_class_weight('balanced', classes=np.unique(df['sarcasm_encoded']), y=df['sarcasm_encoded'])
    hate_weights = torch.tensor(hate_weights, dtype=torch.float).to(device)
    sarcasm_weights = torch.tensor(sarcasm_weights, dtype=torch.float).to(device)
    print(f"Hate Speech Weights: {hate_weights.cpu().numpy()}")
    print(f"Sarcasm Weights: {sarcasm_weights.cpu().numpy()}")
    
    loss_fn_hate = nn.CrossEntropyLoss(weight=hate_weights)
    loss_fn_sarcasm = nn.CrossEntropyLoss(weight=sarcasm_weights)
    loss_fn_emotion = nn.CrossEntropyLoss()

    base_model = AutoModel.from_pretrained(config.MODEL_NAME)
    model = MultiTaskModel(
        base_model=base_model,
        num_hate=len(hate_encoder.classes_),
        num_sarcasm=len(sarcasm_encoder.classes_),
        num_emotion=len(emotion_encoder.classes_)
    )
    model.load_state_dict(torch.load(config.BASELINE_MODEL_PATH))
    model.to(device)
    print("Successfully loaded weights from the baseline model.")

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)

    print("Starting fine-tuning with enhanced settings...")
    for epoch in range(config.EPOCHS):
        print(f"\n--- Fine-Tuning Epoch {epoch + 1}/{config.EPOCHS} ---")
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc="Fine-Tuning"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            hate_labels = batch['hate_labels'].to(device)
            sarcasm_labels = batch['sarcasm_labels'].to(device)
            emotion_labels = batch['emotion_labels'].to(device)

            optimizer.zero_grad()
            hate_logits, sarcasm_logits, emotion_logits = model(input_ids, attention_mask)
            
            loss_hate = loss_fn_hate(hate_logits, hate_labels)
            loss_sarcasm = loss_fn_sarcasm(sarcasm_logits, sarcasm_labels)
            loss_emotion = loss_fn_emotion(emotion_logits, emotion_labels)
            
            task_weights = {'hate': 1.5, 'sarcasm': 1.2, 'emotion': 1.0}
            loss = (task_weights['hate'] * loss_hate + 
                    task_weights['sarcasm'] * loss_sarcasm + 
                    task_weights['emotion'] * loss_emotion)
            
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch + 1} | Average Fine-Tuning Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), config.FINAL_MODEL_PATH)

if __name__ == "__main__":
    print("\n--- STEP 3B: MODEL ENHANCEMENT & FINE-TUNING STARTING ---")
    enhance_model()
    print("\n--- SCRIPT COMPLETE ---")
    if os.path.exists(Config.FINAL_MODEL_PATH):
        print(f" Model enhancement and fine-tuning are complete.")
        print(f" Final, optimized model saved to: {os.path.abspath(Config.FINAL_MODEL_PATH)}")
        print("\n--- NEXT STEP ---")
        print(" Congratulations! You now have a production-ready model.")
        print("   The final step is to build the FastAPI application to serve this model.")
    else:
        print(" Script finished, but no final model file was created.")