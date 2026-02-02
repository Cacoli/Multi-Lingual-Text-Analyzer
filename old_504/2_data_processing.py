# 2_data_processing.py
#
# --- GOAL ---
# 1. Convert emojis into descriptive text.
# 2. Remove URLs, @mentions, and #hashtags.
# 3. Clean up extra whitespace.
# 4. PRESERVE Hinglish, Tanglish, and other language characters.

import pandas as pd
import re
import emoji
import os
from tqdm import tqdm

# --- Configuration ---
INPUT_FILENAME = "raw_synthetic_data.csv"
OUTPUT_FILENAME = "final_processed_dataset.csv"

def clean_text_multilingual(text: str) -> str:
    """A robust function to clean tweet-like text while preserving all languages."""
    if not isinstance(text, str):
        return ""
    
    # 1. Convert emojis to descriptive text 
    text = emoji.demojize(text, delimiters=(":", ":"))
    
    # 2. Remove URLs, @mentions, and #hashtags
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    
    # 3. Standardize whitespace
    text = re.sub(r'\s+', ' ', text).strip()    
    return text

def process_data():
    """Main function to read, process, and save the data."""
    print(f"Reading raw data from '{INPUT_FILENAME}'...")
    try:
        df = pd.read_csv(INPUT_FILENAME)
    except FileNotFoundError:
        print(f"ERROR: The input file '{INPUT_FILENAME}' was not found.")
        return

    print("Cleaning and processing text with MULTILINGUAL support...")
    
    tqdm.pandas(desc="Cleaning tweets")
    df['text'] = df['text'].progress_apply(clean_text_multilingual)
    
    initial_rows = len(df)
    
    # Drop any rows that have become empty after cleaning
    df.dropna(subset=['text'], inplace=True)
    df = df[df['text'] != '']
    
    final_rows = len(df)
    rows_dropped = initial_rows - final_rows
    
    print(f"Dropped {rows_dropped} rows that were empty after cleaning.")
    
    df.to_csv(OUTPUT_FILENAME, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    print("\n--- STEP 2: DATA PROCESSING (Corrected Multilingual Version) ---")
    process_data()
    print("\n--- SCRIPT COMPLETE ---")
    if os.path.exists(OUTPUT_FILENAME):
        print(f"Successfully cleaned the data while preserving all languages.")
        print(f"Final processed data saved to: {os.path.abspath(OUTPUT_FILENAME)}")
        print("\n--- NEXT STEP ---")
        print("Please inspect the new `final_processed_dataset.csv`. You should see the Hinglish and Tanglish text is now preserved.")
        print("Once you're happy with it, we can proceed to Step 3: Model Training.")
    else:
        print("Script finished, but no output file was created.")