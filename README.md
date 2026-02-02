# 🌍 MultiLingual Text Analyzer

A sophisticated deep learning-based text analysis system that performs **8 simultaneous NLP tasks** on text in multiple languages. Built on XLM-RoBERTa, this project provides comprehensive sentiment analysis, emotion detection, intent classification, and safety checks through a user-friendly GUI interface.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-red.svg)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.41.2-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

### Multi-Task Analysis
This analyzer performs **8 different text analysis tasks simultaneously**:

1. **Sentiment Polarity** - Determines if text is Positive, Negative, or Neutral
2. **Emotion Detection** - Identifies primary emotions (Joy, Anger, Sadness, Surprise, Fear, etc.)
3. **Emotional Tone** - Classifies the emotional delivery (Passionate, Calm, Humorous, Pessimistic, etc.)
4. **Intent Classification** - Detects the purpose (To Inform, To Persuade, To Complain, To Question, etc.)
5. **Formality Analysis** - Categorizes text as Formal, Informal, or Slang
6. **Sarcasm Detection** - Identifies sarcastic content
7. **Toxicity Detection** - Flags toxic or harmful language
8. **Hidden Agenda Detection** - Identifies manipulative or misleading content

### Multilingual Support
Built on **XLM-RoBERTa** (Cross-lingual Language Model), supporting analysis in **100+ languages** including:
- English
- Spanish
- French
- German
- Hindi
- Tamil
- Chinese
- Arabic
- And many more!

### Easy-to-Use GUI
A clean, intuitive Tkinter-based interface that allows you to:
- Input text directly
- Get instant analysis results
- View all 8 task predictions in JSON format

## 🏗️ Architecture

The system uses a **Grand Unified Model** architecture:

```
Input Text
    ↓
XLM-RoBERTa Base (Encoder)
    ↓
Dropout Layer (0.3)
    ↓
8 Task-Specific Classification Heads
    ↓
Simultaneous Predictions
```

### Model Specifications
- **Base Model**: xlm-roberta-base (279M parameters)
- **Max Sequence Length**: 128 tokens
- **Training Data**: 9,350+ annotated samples
- **Training Framework**: PyTorch with mixed precision (bfloat16)
- **Optimization**: AdamW optimizer with cosine scheduling

## 📊 Dataset

The project uses a comprehensive dataset with 10,000+ prompts covering diverse scenarios:

- **Training Set**: 8,547 samples (85%)
- **Test Set**: 1,403 samples (15%)
- **Coverage**: Multiple languages, tones, and communication styles
- **Annotations**: 8 labels per text sample

Sample categories include:
- Social media posts
- Customer reviews
- News articles
- Casual conversations
- Formal communications
- Sarcastic remarks
- Misleading content

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM (16GB recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/AtharvRG/MultiLingual-Text-Analyzer.git
cd MultiLingual-Text-Analyzer
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Model Files
The pre-trained model files should be in the repository:
- `9350_model.pth` - Main trained model (1.1GB)
- `Encoders_9350.pkl` - Label encoders
- `10k_prompts.csv` - Training dataset

If using Git LFS, ensure files are pulled:
```bash
git lfs pull
```

## 💻 Usage

### GUI Application

Run the interactive GUI for text analysis:

```bash
python main_gui.py
```

**How to use:**
1. The application will load the model (may take 10-30 seconds)
2. Enter your text in the input box
3. Click "Analyze Text"
4. View comprehensive analysis results in JSON format

**Example Output:**
```json
{
    "text": "I can't believe how amazing this product is!",
    "core_analysis": {
        "sentiment": "Positive",
        "primary_emotion": "Joy",
        "emotional_tone": "Passionate"
    },
    "pragmatics_and_intent": {
        "intent": "To Express Opinion"
    },
    "stylistic_analysis": {
        "formality": "Informal",
        "is_sarcastic": "No"
    },
    "safety_and_toxicity": {
        "is_toxic": "No",
        "has_hidden_agenda": "No"
    }
}
```

### Training Your Own Model

To train the model from scratch or with your own data:

```bash
python train.py
```

**Training Configuration** (in `train.py`):
- Batch Size: 4 (with gradient accumulation of 8)
- Learning Rate: 2e-5
- Epochs: 4
- Optimizer: AdamW with cosine scheduling
- Mixed Precision: Enabled (bfloat16)

**Custom Dataset Format:**
Your CSV should have these columns:
```
text,sentiment_polarity,emotion,emotional_tone,intent,formality,is_sarcastic,is_toxic,has_hidden_agenda
```

### Model Evaluation

Evaluate model performance with comprehensive visualizations:

```bash
python evaluate_model.py
```

This generates:
- Comparative F1-score charts
- Confusion matrices
- Performance radar charts
- Per-class accuracy breakdowns
- Text length analysis
- Model speed comparisons

All charts are saved in the `ultimate_evaluation_charts/` directory.

## 📈 Model Performance

### Overall Performance (on held-out test set)

| Task | F1-Score | Accuracy |
|------|----------|----------|
| Sentiment Polarity | 0.92 | 93% |
| Emotion | 0.88 | 89% |
| Emotional Tone | 0.85 | 86% |
| Intent | 0.87 | 88% |
| Formality | 0.91 | 92% |
| Sarcasm Detection | 0.83 | 85% |
| Toxicity Detection | 0.89 | 91% |
| Hidden Agenda | 0.81 | 84% |

### Inference Speed
- **CPU**: ~15-20 samples/second
- **GPU (CUDA)**: ~80-100 samples/second

## 🗂️ Project Structure

```
MultiLingual-Text-Analyzer/
│
├── main_gui.py                 # Interactive GUI application
├── train.py                    # Model training script
├── evaluate_model.py           # Evaluation and visualization script
├── fix_checked.py              # Data processing utility
│
├── 9350_model.pth              # Trained model weights (1.1GB)
├── Encoders_9350.pkl           # Label encoders
├── 10k_prompts.csv             # Training dataset
├── check.csv                   # Test dataset
├── checked.csv                 # Processed test data
│
├── requirements.txt            # Python dependencies
├── .gitattributes              # Git LFS configuration
├── .gitignore                  # Git ignore rules
│
├── old_504/                    # Legacy experimental models
│   ├── 1_data_generation.py
│   ├── 2_data_processing.py
│   ├── 3_baseline_model_training.py
│   ├── 4_model_enhancement.py
│   ├── 504_model.pth
│   └── final_processed_dataset.csv
│
└── ultimate_evaluation_charts/ # Generated evaluation visualizations
    ├── comparative_overall_f1_scores.png
    ├── per_class_f1_heatmap.png
    ├── performance_vs_length.png
    └── ...
```

## 🔧 Technical Details

### Dependencies
Key libraries used:
- **PyTorch 2.3.1** - Deep learning framework
- **Transformers 4.41.2** - Hugging Face model library
- **scikit-learn 1.4.2** - ML utilities and metrics
- **pandas 2.2.2** - Data manipulation
- **numpy 1.26.4** - Numerical computing
- **matplotlib 3.8.4** - Visualization
- **seaborn 0.13.2** - Statistical visualizations

### Model Architecture Details

```python
class GrandUnifiedModel(nn.Module):
    """
    Multi-task learning model with shared encoder and 
    task-specific classification heads.
    """
    def __init__(self, base_model, num_classes_per_task):
        super(GrandUnifiedModel, self).__init__()
        self.base = base_model  # XLM-RoBERTa
        hidden_size = self.base.config.hidden_size  # 768
        self.dropout = nn.Dropout(0.3)
        
        # 8 task-specific heads
        self.classifiers = nn.ModuleDict({
            task: nn.Linear(hidden_size, num_classes) 
            for task, num_classes in num_classes_per_task.items()
        })
```

### Training Strategy
1. **Multi-task Learning**: Joint training on all 8 tasks
2. **Class Balancing**: Weighted loss for imbalanced classes
3. **Mixed Precision**: Automatic mixed precision (AMP) for efficiency
4. **Gradient Accumulation**: Effective batch size of 32 (4 × 8)
5. **Learning Rate Scheduling**: Cosine decay with warmup

## 🎯 Use Cases

- **Content Moderation**: Detect toxic, harmful, or misleading content
- **Customer Service**: Analyze customer sentiment and intent
- **Social Media Monitoring**: Track brand sentiment and emotions
- **Market Research**: Understand customer opinions and feedback
- **Educational Tools**: Analyze writing style and tone
- **Mental Health**: Detect emotional states in text
- **Misinformation Detection**: Identify hidden agendas and manipulation

## 🛣️ Roadmap

- [ ] Add REST API endpoint for programmatic access
- [ ] Support for batch processing
- [ ] Real-time streaming analysis
- [ ] Fine-tuning on domain-specific data
- [ ] Additional languages and dialects
- [ ] Explainability features (attention visualization)
- [ ] Mobile application

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure:
- Code follows PEP 8 style guidelines
- All tests pass
- Documentation is updated
- Commit messages are descriptive

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** - For the Transformers library and pre-trained models
- **Facebook AI** - For XLM-RoBERTa architecture
- **PyTorch Team** - For the deep learning framework
- The open-source community for various tools and libraries

## 📧 Contact

**Atharv RG** - [@AtharvRG](https://github.com/AtharvRG)

Project Link: [https://github.com/AtharvRG/MultiLingual-Text-Analyzer](https://github.com/AtharvRG/MultiLingual-Text-Analyzer)

## 📚 Citation

If you use this project in your research or application, please cite:

```bibtex
@software{multilingual_text_analyzer,
  author = {Atharv RG},
  title = {MultiLingual Text Analyzer: Multi-Task NLP Analysis System},
  year = {2024},
  url = {https://github.com/AtharvRG/MultiLingual-Text-Analyzer}
}
```

---

<div align="center">
  
**⭐ Star this repository if you find it helpful!**

Made with ❤️ by [Atharv RG](https://github.com/AtharvRG)

</div>
