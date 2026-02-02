# =============================================================================
# FINAL, ROBUST TRAINING & EVALUATION SCRIPT
#
# STRATEGY:
#   - Trains the model on the new, leakage-proof dataset.
#   - Immediately evaluates it on a held-out test set.
#   - Fixes the classification_report crash by explicitly defining labels.
# =============================================================================

import pandas as pd
import torch, torch.nn as nn, numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from tqdm import tqdm
import os, pickle

class Config:
    INPUT_FILENAME = "10k_prompts.csv"
    MODEL_NAME = "xlm-roberta-base"
    OUTPUT_MODEL_PATH = "9350_model.pth"
    ENCODERS_PATH = "Encoders_9350.pkl"
    LABEL_COLUMNS = ["sentiment_polarity", "emotion", "emotional_tone", "intent", "formality", "is_sarcastic", "is_toxic", "has_hidden_agenda"]
    MAX_LEN, BATCH_SIZE, GRAD_ACCUM, EPOCHS, LR = 128, 4, 8, 4, 2e-5

class MultiTaskDataset(Dataset):
    def __init__(self, texts, labels_dict, tokenizer, max_len): self.texts, self.labels, self.tokenizer, self.max_len = texts, labels_dict, tokenizer, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, item):
        encoding = self.tokenizer(str(self.texts[item]), max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        labels = {key: torch.tensor(self.labels[key][item], dtype=torch.long) for key in self.labels}
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), **labels}
class GrandUnifiedModel(nn.Module):
    def __init__(self, base, num_classes):
        super().__init__(); self.base = base; self.dropout = nn.Dropout(0.3)
        self.classifiers = nn.ModuleDict({task: nn.Linear(base.config.hidden_size, n) for task, n in num_classes.items()})
    def forward(self, input_ids, attention_mask):
        pooled = self.dropout(self.base(input_ids=input_ids, attention_mask=attention_mask).pooler_output)
        return {task: clf(pooled) for task, clf in self.classifiers.items()}

def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(config.INPUT_FILENAME).dropna()
    
    # --- TRAINING ---
    label_encoders = {col: LabelEncoder().fit(df[col]) for col in config.LABEL_COLUMNS}
    for col, le in label_encoders.items(): df[f"{col}_encoded"] = le.transform(df[col])
    with open(config.ENCODERS_PATH, 'wb') as f: pickle.dump(label_encoders, f)

    df_train, df_test = train_test_split(df, test_size=0.15, random_state=42) # Using a 15% test set
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    train_labels = {col: df_train[f"{col}_encoded"].to_numpy() for col in config.LABEL_COLUMNS}
    train_loader = DataLoader(MultiTaskDataset(df_train.text.to_numpy(), train_labels, tokenizer, config.MAX_LEN), batch_size=config.BATCH_SIZE, shuffle=True)
    
    num_classes = {col: len(le.classes_) for col, le in label_encoders.items()}
    model = GrandUnifiedModel(AutoModel.from_pretrained(config.MODEL_NAME), num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=config.LR)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * config.EPOCHS // config.GRAD_ACCUM)
    loss_fns = {c: nn.CrossEntropyLoss(weight=torch.tensor(compute_class_weight('balanced', classes=np.unique(df[f"{c}_encoded"]), y=df[f"{c}_encoded"]), dtype=torch.float).to(device)) if len(label_encoders[c].classes_) <= 3 else nn.CrossEntropyLoss() for c in config.LABEL_COLUMNS}
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.EPOCHS):
        model.train(); total_loss = 0
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
                loss = sum(loss_fns[task](logits[task], batch[task].to(device)) for task in config.LABEL_COLUMNS) / config.GRAD_ACCUM
            scaler.scale(loss).backward(); total_loss += loss.item()
            if (i + 1) % config.GRAD_ACCUM == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad()
        print(f"Epoch {epoch + 1} | Avg Loss: {total_loss / len(train_loader) * config.GRAD_ACCUM:.4f}")

    torch.save(model.state_dict(), config.OUTPUT_MODEL_PATH)
    print(f"Final Model saved to: {config.OUTPUT_MODEL_PATH}")

    # --- EVALUATION ---
    print("\n--- Starting Evaluation on Unseen Test Set ---")
    model.eval()
    test_labels = {col: df_test[f"{col}_encoded"].to_numpy() for col in config.LABEL_COLUMNS}
    test_loader = DataLoader(MultiTaskDataset(df_test.text.to_numpy(), test_labels, tokenizer, config.MAX_LEN), batch_size=config.BATCH_SIZE)
    
    y_true, y_pred = {task: [] for task in config.LABEL_COLUMNS}, {task: [] for task in config.LABEL_COLUMNS}
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            for task in config.LABEL_COLUMNS:
                y_pred[task].extend(torch.argmax(logits[task], dim=1).cpu().numpy())
                y_true[task].extend(batch[task].cpu().numpy())

    print("\n--- FINAL MODEL PERFORMANCE REPORT ---")
    for task in config.LABEL_COLUMNS:
        print(f"\n==================== TASK: {task.upper()} ====================")
        true_labels = y_true[task]
        pred_labels = y_pred[task]
        report_labels = np.unique(np.concatenate((true_labels, pred_labels)))
        target_names = label_encoders[task].inverse_transform(report_labels)
        print(classification_report(true_labels, pred_labels, labels=report_labels, target_names=target_names, zero_division=0))

if __name__ == "__main__":
    main()