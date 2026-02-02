# =============================================================================
# DEEP TEXT ANALYZER - GUI APPLICATION
#
# A simple Tkinter-based user interface to interact with the trained model.
# This is a self-contained script that loads the model and provides a UI.
# =============================================================================

import torch
import torch.nn as nn
import pandas as pd
import pickle
from transformers import AutoTokenizer, AutoModel
import tkinter as tk
from tkinter import scrolledtext
import json

# --- CONFIGURATION & SETUP ---
class Config:
    MODEL_PATH = "9350_model.pth"
    TOKENIZER_NAME = "xlm-roberta-base"
    ENCODERS_PATH = "Encoders_9350.pkl"
    MAX_LEN = 128

# --- Define Model Architecture ---
class GrandUnifiedModel(nn.Module):
    def __init__(self, base_model, num_classes_per_task):
        super(GrandUnifiedModel, self).__init__()
        self.base = base_model
        hidden_size = self.base.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.classifiers = nn.ModuleDict({task: nn.Linear(hidden_size, num_classes) for task, num_classes in num_classes_per_task.items()})
    def forward(self, input_ids, attention_mask):
        pooled_output = self.base(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        pooled_output = self.dropout(pooled_output)
        return {task: classifier(pooled_output) for task, classifier in self.classifiers.items()}

# --- Global variables to hold the loaded model and assets ---
device = None
tokenizer = None
label_encoders = None
model = None

def load_all_assets():
    """Loads the tokenizer, encoders, and the trained model into memory."""
    global device, tokenizer, label_encoders, model
    
    print("Loading all assets... This may take a moment.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME)

    with open(Config.ENCODERS_PATH, 'rb') as f:
        label_encoders = pickle.load(f)
    print("Label encoders loaded.")

    num_classes_per_task = {col: len(le.classes_) for col, le in label_encoders.items()}
    base_model = AutoModel.from_pretrained(Config.TOKENIZER_NAME)
    model = GrandUnifiedModel(base_model, num_classes_per_task)

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("FINAL Model loaded successfully!")

def predict_text(text: str):
    """Takes raw text and returns a dictionary of model predictions."""
    if device is None or tokenizer is None or label_encoders is None or model is None:
        return {"error": "Model assets are not loaded."}

    encoding = tokenizer(text, add_special_tokens=True, max_length=Config.MAX_LEN, padding='max_length', truncation=True, return_tensors='pt')
    
    with torch.no_grad():
        logits_dict = model(encoding['input_ids'].to(device), encoding['attention_mask'].to(device))
    
    predictions = {
        task: label_encoders[task].inverse_transform([torch.argmax(logits, dim=1).item()])[0]
        for task, logits in logits_dict.items()
    }
    
    return {
        "text": text,
        "core_analysis": {"sentiment": predictions.get("sentiment_polarity"), "primary_emotion": predictions.get("emotion"), "emotional_tone": predictions.get("emotional_tone")},
        "pragmatics_and_intent": {"intent": predictions.get("intent")},
        "stylistic_analysis": {"formality": predictions.get("formality"), "is_sarcastic": predictions.get("is_sarcastic")},
        "safety_and_toxicity": {"is_toxic": predictions.get("is_toxic"), "has_hidden_agenda": predictions.get("has_hidden_agenda")}
    }

# --- GUI Application Logic ---

def on_analyze_click():
    """Handles the button click event."""
    input_text = text_input.get("1.0", tk.END).strip()
    if not input_text:
        result_text.configure(state='normal')
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.INSERT, "Please enter some text to analyze.")
        result_text.configure(state='disabled')
        return

    # Perform prediction
    predictions = predict_text(input_text)
    
    # Format the output as a pretty-printed JSON string
    formatted_result = json.dumps(predictions, indent=4)
    
    # Update the result text box
    result_text.configure(state='normal') # Enable writing
    result_text.delete("1.0", tk.END)
    result_text.insert(tk.INSERT, formatted_result)
    result_text.configure(state='disabled') # Disable writing

# --- Main Application Window ---
if __name__ == "__main__":
    # Load model first
    load_all_assets()
    
    # Create the main window
    window = tk.Tk()
    window.title("Deep Text Analyzer")
    window.geometry("700x600")

    # Create and configure widgets
    main_frame = tk.Frame(window, padx=10, pady=10)
    main_frame.pack(fill="both", expand=True)

    label_input = tk.Label(main_frame, text="Enter Text to Analyze:", font=("Helvetica", 12))
    label_input.pack(anchor="w")

    text_input = scrolledtext.ScrolledText(main_frame, height=10, font=("Arial", 10))
    text_input.pack(fill="x", pady=5)

    analyze_button = tk.Button(main_frame, text="Analyze Text", command=on_analyze_click, font=("Helvetica", 12, "bold"), bg="#4CAF50", fg="white")
    analyze_button.pack(pady=10)

    label_result = tk.Label(main_frame, text="Analysis Results:", font=("Helvetica", 12))
    label_result.pack(anchor="w")

    result_text = scrolledtext.ScrolledText(main_frame, height=20, font=("Courier New", 10), state='disabled', bg="#f0f0f0")
    result_text.pack(fill="both", expand=True, pady=5)

    # Start the GUI event loop
    window.mainloop()