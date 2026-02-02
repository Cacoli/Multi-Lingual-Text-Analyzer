# old_504/1_data_generation.py

import pandas as pd
import torch
from transformers import pipeline, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os

# --- CONFIGURATION ---
class Config:
    MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
    OUTPUT_FILENAME = "raw_synthetic_data.csv"
    
    BATCH_SIZE = 8
    
    TOPICS = ["a new tax policy", "political corruption", "a recent election result", 
              "a new infrastructure project", "a politician's public speech", "freedom of speech",
              "a healthcare reform bill", "environmental regulations"]
    
    LANGUAGES = [
        {"name": "English", "code": "en"},
        {"name": "Hinglish (Hindi+English)", "code": "hi_en"},
        {"name": "Tanglish (Tamil+English)", "code": "ta_en"}
    ]
    
    SENTIMENTS = [
        {"tone": "genuinely happy and supportive", "hate_speech": "No", "sarcasm": "No", "emotion": "Joy"},
        {"tone": "angry and frustrated", "hate_speech": "No", "sarcasm": "No", "emotion": "Anger"},
        {"tone": "sad and disappointed", "hate_speech": "No", "sarcasm": "No", "emotion": "Sadness"},
        {"tone": "sarcastic and mocking", "hate_speech": "No", "sarcasm": "Yes", "emotion": "Anger"},
        {"tone": "hateful and aggressive", "hate_speech": "Yes", "sarcasm": "No", "emotion": "Anger"},
        {"tone": "fearful and concerned", "hate_speech": "No", "sarcasm": "No", "emotion": "Fear"},
        {"tone": "neutral and observational", "hate_speech": "No", "sarcasm": "No", "emotion": "Neutral"},
    ]

    SAMPLES_PER_COMBINATION = 3
    GENERATION_PARAMS = {
        "max_new_tokens": 80,
        "do_sample": True,
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.95,
    }

def create_prompt_blueprint(config):
    """Generates a list of structured prompts, repeating them for the desired number of samples."""
    blueprint = []
    base_prompts = []
    for topic in config.TOPICS:
        for lang in config.LANGUAGES:
            for sentiment in config.SENTIMENTS:
                prompt_text = f"Generate a short, realistic tweet in {lang['name']} that is {sentiment['tone']} about {topic}."
                base_prompts.append({"prompt": prompt_text, **sentiment})
    
    for prompt in base_prompts:
        for _ in range(config.SAMPLES_PER_COMBINATION):
            blueprint.append(prompt)
            
    return blueprint

def generate_data(config):
    """Main function to generate the dataset using a batched approach."""
    
    # Device Check and Model Loading
    if not torch.cuda.is_available():
        print("🚨 WARNING: CUDA not available...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device} (PyTorch: {torch.__version__})")

    blueprint = create_prompt_blueprint(config)
    print(f"  Created a blueprint with {len(blueprint)} total prompts to generate.")

    try:
        print(f"Loading model '{config.MODEL_NAME}' with 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME, quantization_config=quantization_config, device_map="auto")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        print("  Model and pipeline created successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # --- BATCHED GENERATION LOGIC ---
    # 1. Prepare all prompts in the format the model expects
    all_prompts_for_model = []
    for item in tqdm(blueprint, desc="Preparing prompts"):
        messages = [{"role": "system", "content": "You are a helpful assistant that generates realistic-sounding tweets."},
                    {"role": "user", "content": item['prompt']}]
        prompt_text = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_prompts_for_model.append(prompt_text)

    # 2. Run the pipeline ONCE on the entire list of prompts
    print(f"\n🚀 Starting batched generation with a batch size of {config.BATCH_SIZE}...")
    outputs = pipe(all_prompts_for_model, **config.GENERATION_PARAMS, batch_size=config.BATCH_SIZE)
    print("  Batched generation complete.")

    # 3. Process the results
    results = []
    # Use zip to match the original blueprint data (labels) with the generated output
    for item, generated_output in tqdm(zip(blueprint, outputs), total=len(blueprint), desc="Processing results"):
        # The output from the pipeline is a list, usually containing one dictionary
        if generated_output and len(generated_output) > 0:
            generated_text = generated_output[0]["generated_text"].split("<|assistant|>")[-1].strip().replace('"', '')
            if len(generated_text) > 20:
                results.append({
                    "text": generated_text,
                    "hate_speech": item["hate_speech"],
                    "sarcasm": item["sarcasm"],
                    "emotion": item["emotion"],
                })

    # Save to CSV
    if not results:
        print("No results were generated. Please check for errors above.")
        return
        
    df = pd.DataFrame(results)
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(config.OUTPUT_FILENAME, index=False, encoding='utf-8-sig')

def main():
    print("--- STEP 1: AUTOMATIC & DIVERSE DATA GENERATION (BATCHED) ---")
    config = Config()
    generate_data(config)
    
    print("\n--- SCRIPT COMPLETE ---")
    if os.path.exists(Config.OUTPUT_FILENAME):
        final_count = len(pd.read_csv(Config.OUTPUT_FILENAME))
        print(f"  Successfully generated and saved {final_count} tweets.")
        print(f"  Data saved to: {os.path.abspath(Config.OUTPUT_FILENAME)}")
        print("\n--- NEXT STEP ---")
        print("🗣️  KUMAR, your turn! The raw dataset is ready for processing.")
    else:
        print("Script finished, but no output file was created.")

if __name__ == "__main__":
    main()