# fix_my_csv.py
#
# GOAL: To rescue a corrupted CSV file where commas within the 'text'
#       column have broken the file structure.

import pandas as pd
from tqdm import tqdm
import os

# --- CONFIGURATION ---
INPUT_FILENAME = "checked.csv"  
OUTPUT_FILENAME = "check.csv"

# The exact header of our CSV file.
HEADER = [
    "text", "sentiment_polarity", "emotion", "emotional_tone", 
    "intent", "formality", "is_sarcastic", "is_toxic", "has_hidden_agenda"
]

# The number of columns that are NOT the 'text' column.
NUM_LABEL_COLUMNS = len(HEADER) - 1

def rescue_csv():
    print(f"Starting rescue operation on '{INPUT_FILENAME}'...")
    
    corrected_rows = []
    
    try:
        with open(INPUT_FILENAME, 'r', encoding='utf-8') as f:
            # Read all lines, skipping the original (and likely broken) header
            lines = f.readlines()[1:]
            
            for line in tqdm(lines, desc="Reconstructing rows"):
                line = line.strip()
                if not line:
                    continue

                # Split the entire line by every comma
                parts = line.split(',')
                # Ensure we have enough parts to reconstruct the row
                if len(parts) >= len(HEADER):
                    text_parts = parts[:-NUM_LABEL_COLUMNS]
                    label_parts = parts[-NUM_LABEL_COLUMNS:]
                    
                    # Rejoin the text parts with the comma they were split by
                    reconstructed_text = ",".join(text_parts).strip('"')
                    
                    # Create the new, clean row
                    new_row = [reconstructed_text] + label_parts
                    corrected_rows.append(new_row)
                else:
                    print(f"  - WARNING: Skipping malformed line: {line[:50]}...")

    except FileNotFoundError:
        print(f"❌ ERROR: File not found! Make sure your data file is named '{INPUT_FILENAME}'.")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    if not corrected_rows:
        print("❌ No rows were successfully reconstructed. Please check the input file.")
        return

    # Create a new, clean DataFrame and save it
    df_corrected = pd.DataFrame(corrected_rows, columns=HEADER)
    df_corrected.to_csv(OUTPUT_FILENAME, index=False, encoding='utf-8-sig')

    print("\n--- RESCUE COMPLETE ---")
    print(f"Successfully reconstructed {len(df_corrected)} rows.")
    print(f"Your clean, final dataset has been saved to: {os.path.abspath(OUTPUT_FILENAME)}")

if __name__ == "__main__":
    rescue_csv()