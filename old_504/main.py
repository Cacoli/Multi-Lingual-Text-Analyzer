# main.py
#
# --- GOAL ---
# This script creates a FastAPI web server to serve our trained model.
# It has two endpoints: one for text input and one for image uploads.

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from pydantic import BaseModel
import os
import io

# --- FastAPI Imports ---
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware # To allow web pages to talk to our API

# --- OCR Imports ---
import pytesseract
from PIL import Image

# --- CONFIGURATION & SETUP ---

# IMPORTANT: Update this path to where you installed Tesseract OCR
# Example for Windows. For Linux/macOS it might be '/usr/bin/tesseract'
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except Exception:
    print("Warning: Tesseract path not found or invalid. OCR endpoint will not work.")
    print("Please install Tesseract and update the path in main.py")


class Config:
    MODEL_PATH = "final_model.pth"
    TOKENIZER_NAME = "xlm-roberta-base"
    DATASET_PATH = "final_processed_dataset.csv" # We need this to refit the encoders
    MAX_LEN = 128

# --- 1. Define Model Architecture (must be identical to training script) ---
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

# --- 2. Load all assets when the server starts ---
print("Loading model and assets... This may take a moment.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME)

# Load the label encoders and fit them on our training data so they know the classes
df = pd.read_csv(Config.DATASET_PATH).dropna()
hate_encoder = LabelEncoder().fit(df['hate_speech'])
sarcasm_encoder = LabelEncoder().fit(df['sarcasm'])
emotion_encoder = LabelEncoder().fit(df['emotion'])

# Load the model architecture
base_model = AutoModel.from_pretrained(Config.TOKENIZER_NAME)
model = MultiTaskModel(
    base_model=base_model,
    num_hate=len(hate_encoder.classes_),
    num_sarcasm=len(sarcasm_encoder.classes_),
    num_emotion=len(emotion_encoder.classes_)
)
# Load the trained weights
model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
model.to(device)
model.eval() # Set model to evaluation mode
print("✅ Model and assets loaded successfully!")


# --- 3. Prediction Function ---
def predict_text(text: str):
    """Takes raw text and returns model predictions."""
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=Config.MAX_LEN,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        hate_logits, sarcasm_logits, emotion_logits = model(input_ids, attention_mask)

    # Convert logits to probabilities and then to class predictions
    hate_pred = hate_encoder.inverse_transform([torch.argmax(hate_logits, dim=1).item()])[0]
    sarcasm_pred = sarcasm_encoder.inverse_transform([torch.argmax(sarcasm_logits, dim=1).item()])[0]
    emotion_pred = emotion_encoder.inverse_transform([torch.argmax(emotion_logits, dim=1).item()])[0]

    return {
        "text": text,
        "hate_speech_prediction": hate_pred,
        "sarcasm_prediction": sarcasm_pred,
        "emotion_prediction": emotion_pred
    }

# --- 4. FastAPI App and Endpoints ---
app = FastAPI(title="Multilingual Tweet Analyzer API")

# Add CORS middleware to allow requests from any origin (e.g., a React frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the request body for the text endpoint
class TextIn(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Tweet Analyzer API. Go to /docs to see the endpoints."}

@app.post("/predict/text")
def predict_from_text(request: TextIn):
    """Receives raw text and returns predictions."""
    try:
        predictions = predict_text(request.text)
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/image")
async def predict_from_image(file: UploadFile = File(...)):
    """Receives an image, performs OCR, and returns predictions."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Read image content
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Perform OCR
        extracted_text = pytesseract.image_to_string(image)
        
        if not extracted_text.strip():
            return {"message": "OCR could not detect any text in the image."}
            
        # Get predictions on the extracted text
        predictions = predict_text(extracted_text)
        return predictions

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")