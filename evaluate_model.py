# =============================================================================
# THE ULTIMATE VISUAL EVALUATION SCRIPT (COMPLETE & UNABRIDGED)
#
# Generates a comprehensive suite of 11 comparative charts, matrices, and visuals.
# This script is self-contained and includes all necessary code.
# =============================================================================

import pandas as pd
import torch, torch.nn as nn, numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm
import os, pickle, time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
# Set a consistent, professional style for all plots
sns.set_theme(style="whitegrid")

class Config:
    # --- IMPORTANT: UPDATE THESE FILENAMES TO MATCH YOUR FILES ---
    PRELIM_MODEL_PATH = "504_model.pth"
    PRELIM_DATASET_PATH = "final_processed_dataset.csv" # The SMALL dataset
    FINAL_MODEL_PATH = "9350_model.pth"
    FINAL_ENCODERS_PATH = "Encoders_9350.pkl"
    TEST_DATA_FILENAME = "check.csv"
    
    MODEL_NAME = "xlm-roberta-base"
    RESULTS_DIR = "ultimate_evaluation_charts" # Folder to save all visuals
    
    PRELIM_LABEL_COLUMNS = ["hate_speech", "sarcasm", "emotion"]
    FINAL_LABEL_COLUMNS = ["sentiment_polarity", "emotion", "emotional_tone", "intent", "formality", "is_sarcastic", "is_toxic", "has_hidden_agenda"]
    
    KEY_TASK_FOR_DETAIL_CHART = "sarcasm"
    KEY_TASK_FOR_ACCURACY_CHART = "emotion"
    MAX_LEN, BATCH_SIZE = 128, 16

# --- DATASET & MODEL CLASSES ---
class MultiTaskDataset(Dataset):
    def __init__(self, texts, labels_dict, tokenizer, max_len, tasks_to_load):
        self.texts, self.labels, self.tokenizer, self.max_len, self.tasks = texts, labels_dict, tokenizer, max_len, tasks_to_load
    def __len__(self): return len(self.texts)
    def __getitem__(self, item):
        encoding = self.tokenizer(str(self.texts[item]), max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        labels = {key: torch.tensor(self.labels.get(key)[item], dtype=torch.long) for key in self.tasks if key in self.labels}
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), **labels}

class PreliminaryModel(nn.Module):
    def __init__(self, base, num_classes):
        super().__init__()
        self.base = base
        self.dropout = nn.Dropout(0.3)
        self.hate_classifier = nn.Linear(base.config.hidden_size, num_classes['hate_speech'])
        self.sarcasm_classifier = nn.Linear(base.config.hidden_size, num_classes['sarcasm'])
        self.emotion_classifier = nn.Linear(base.config.hidden_size, num_classes['emotion'])
    def forward(self, input_ids, attention_mask):
        pooled = self.dropout(self.base(input_ids=input_ids, attention_mask=attention_mask).pooler_output)
        return {
            'hate_speech': self.hate_classifier(pooled), 
            'sarcasm': self.sarcasm_classifier(pooled), 
            'emotion': self.emotion_classifier(pooled)
        }

class FinalModel(nn.Module):
    def __init__(self, base, num_classes):
        super().__init__()
        self.base = base
        self.dropout = nn.Dropout(0.3)
        self.classifiers = nn.ModuleDict({task: nn.Linear(base.config.hidden_size, n) for task, n in num_classes.items()})
    def forward(self, input_ids, attention_mask):
        pooled = self.dropout(self.base(input_ids=input_ids, attention_mask=attention_mask).pooler_output)
        return {task: clf(pooled) for task, clf in self.classifiers.items()}

def run_evaluation(model, test_loader, tasks, device):
    y_true, y_pred = {task: [] for task in tasks}, {task: [] for task in tasks}
    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            for task in tasks:
                if task in batch and task in logits:
                    y_pred[task].extend(torch.argmax(logits[task], dim=1).cpu().numpy())
                    y_true[task].extend(batch[task].cpu().numpy())
    end_time = time.time()
    inference_speed = len(test_loader.dataset) / (end_time - start_time) if end_time > start_time else 0
    return y_true, y_pred, inference_speed

def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # --- Load all assets ---
    print("\nLoading assets...");
    try:
        df_prelim_train = pd.read_csv(config.PRELIM_DATASET_PATH).dropna()
        prelim_encoders = {c: LabelEncoder().fit(df_prelim_train[c]) for c in config.PRELIM_LABEL_COLUMNS}
        prelim_model = PreliminaryModel(AutoModel.from_pretrained(config.MODEL_NAME), {c: len(le.classes_) for c, le in prelim_encoders.items()})
        prelim_model.load_state_dict(torch.load(config.PRELIM_MODEL_PATH, map_location=device)); prelim_model.to(device); prelim_model.eval()
        
        with open(config.FINAL_ENCODERS_PATH, 'rb') as f: final_encoders = pickle.load(f)
        final_model = FinalModel(AutoModel.from_pretrained(config.MODEL_NAME), {c: len(le.classes_) for c, le in final_encoders.items()})
        final_model.load_state_dict(torch.load(config.FINAL_MODEL_PATH, map_location=device)); final_model.to(device); final_model.eval()
        print("  All model assets loaded successfully.")
    except Exception as e: print(f"❌ ERROR loading assets: {e}"); return

    # --- Prepare test set ---
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    try:
        df_test = pd.read_csv(config.TEST_DATA_FILENAME).dropna()
        all_encoders = {**prelim_encoders, **final_encoders}
        valid_mask = pd.Series([True] * len(df_test))
        for col, le in all_encoders.items():
            if col in df_test.columns: valid_mask &= df_test[col].isin(le.classes_)
        df_test_filtered = df_test[valid_mask].reset_index(drop=True)
        print(f"\nPrepared test set with {len(df_test_filtered)} valid samples.")
        test_labels = {c: le.transform(df_test_filtered[c]) for c, le in all_encoders.items() if c in df_test_filtered.columns}
    except Exception as e: print(f"❌ ERROR processing test file: {e}"); return
    
    all_tasks = sorted(list(set(config.PRELIM_LABEL_COLUMNS) | set(config.FINAL_LABEL_COLUMNS)))
    test_dataset = MultiTaskDataset(df_test_filtered.text.to_numpy(), test_labels, tokenizer, config.MAX_LEN, all_tasks)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)

    # --- Evaluate both models ---
    print("\n--- Evaluating Preliminary Model ---"); prelim_true, prelim_pred, prelim_speed = run_evaluation(prelim_model, test_loader, config.PRELIM_LABEL_COLUMNS, device)
    print("\n--- Evaluating Final Model ---"); final_true, final_pred, final_speed = run_evaluation(final_model, test_loader, config.FINAL_LABEL_COLUMNS, device)

    # --- Generate reports and collect data for visuals ---
    print("\n\n--- HEAD-TO-HEAD PERFORMANCE REPORT ---")
    all_f1_scores, per_class_metrics = [], []
    common_tasks = sorted(list(set(config.PRELIM_LABEL_COLUMNS) & set(config.FINAL_LABEL_COLUMNS)))

    for task in all_tasks:
        print(f"\n\n==================== TASK: {task.upper()} ====================")
        
        # Prelim Model
        print(f"\n--- Preliminary Model (n=504) Results ---")
        if task in config.PRELIM_LABEL_COLUMNS and prelim_true.get(task):
            report_labels = np.unique(np.concatenate((prelim_true[task], prelim_pred[task])))
            target_names = prelim_encoders[task].inverse_transform(report_labels)
            report_dict = classification_report(prelim_true[task], prelim_pred[task], labels=report_labels, target_names=target_names, zero_division=0, output_dict=True)
            print(classification_report(prelim_true[task], prelim_pred[task], labels=report_labels, target_names=target_names, zero_division=0))
            all_f1_scores.append({'Task': task, 'Model': 'Preliminary (n=504)', 'F1-Score': report_dict.get('weighted avg', {}).get('f1-score', 0.0)})
            for label, metrics in report_dict.items():
                if label in prelim_encoders[task].classes_: per_class_metrics.append({'Task': task, 'Class': label, 'Model': 'Preliminary (n=504)', **metrics})
        else:
            print("TASK NOT APPLICABLE"); all_f1_scores.append({'Task': task, 'Model': 'Preliminary (n=504)', 'F1-Score': 0.0})

        # Final Model
        print(f"\n--- Final Model (n=9350) Results ---")
        if task in config.FINAL_LABEL_COLUMNS and final_true.get(task):
            report_labels = np.unique(np.concatenate((final_true[task], final_pred[task])))
            target_names = final_encoders[task].inverse_transform(report_labels)
            report_dict = classification_report(final_true[task], final_pred[task], labels=report_labels, target_names=target_names, zero_division=0, output_dict=True)
            print(classification_report(final_true[task], final_pred[task], labels=report_labels, target_names=target_names, zero_division=0))
            all_f1_scores.append({'Task': task, 'Model': 'Final (n=9350)', 'F1-Score': report_dict.get('weighted avg', {}).get('f1-score', 0.0)})
            for label, metrics in report_dict.items():
                if label in final_encoders[task].classes_: per_class_metrics.append({'Task': task, 'Class': label, 'Model': 'Final (n=9350)', **metrics})
        else:
            print("TASK NOT APPLICABLE"); all_f1_scores.append({'Task': task, 'Model': 'Final (n=9350)', 'F1-Score': 0.0})

    # --- GENERATE ALL VISUALIZATIONS ---
    print("\n\n--- GENERATING VISUALIZATIONS ---")
    
    # VISUAL 1: Overall Comparative F1-Score Bar Chart
    f1_df = pd.DataFrame(all_f1_scores)
    plt.figure(figsize=(12, 8)); sns.barplot(x='F1-Score', y='Task', hue='Model', data=f1_df.sort_values(by=['Task','Model']), palette={'Preliminary (n=504)': 'salmon', 'Final (n=9350)': 'skyblue'})
    plt.title('Comparative Weighted F1-Scores', fontsize=18); plt.xlabel('Weighted F1-Score'); plt.ylabel('Task'); plt.xlim(0, 1.05); plt.legend(title='Model'); plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'comparative_overall_f1_scores.png')); plt.close()
    print("  1. Overall F1-score chart saved.")

    # VISUAL 2: Performance Radar Chart
    radar_df = f1_df.pivot(index='Task', columns='Model', values='F1-Score').fillna(0)
    radar_df_common = radar_df.loc[common_tasks]
    labels = radar_df_common.index.to_list()
    if len(labels) > 0:
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist(); angles += angles[:1]
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        for model_name, color in [('Preliminary (n=504)', 'salmon'), ('Final (n=9350)', 'skyblue')]:
            if model_name in radar_df_common.columns:
                values = radar_df_common[model_name].tolist(); values += values[:1]
                ax.plot(angles, values, color=color, linewidth=2, label=model_name); ax.fill(angles, values, color=color, alpha=0.25)
        ax.set_yticklabels([]); ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
        plt.title('Performance Fingerprint (Common Tasks)', size=20, y=1.1); plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.savefig(os.path.join(config.RESULTS_DIR, 'performance_radar_chart.png')); plt.close()
        print("  2. Performance radar chart saved.")

    # VISUAL 3: Detailed Precision/Recall/F1 Breakdown for Key Task
    detail_df = pd.DataFrame(per_class_metrics)
    detail_df_key_task = detail_df[detail_df['Task'] == config.KEY_TASK_FOR_DETAIL_CHART]
    if not detail_df_key_task.empty and len(detail_df_key_task['Model'].unique()) == 2:
        melted_df = detail_df_key_task.melt(id_vars=['Class', 'Model'], value_vars=['precision', 'recall', 'f1-score'], var_name='Metric', value_name='Score')
        g = sns.catplot(data=melted_df, x='Metric', y='Score', hue='Model', col='Class', kind='bar', palette={'Preliminary (n=504)': 'salmon', 'Final (n=9350)': 'skyblue'}, height=5, aspect=0.8)
        g.fig.suptitle(f'Detailed Metric Breakdown for: {config.KEY_TASK_FOR_DETAIL_CHART.upper()}', y=1.03, fontsize=16)
        g.set_axis_labels("Metric", "Score").set_titles("Class: {col_name}").tight_layout(w_pad=1)
        plt.savefig(os.path.join(config.RESULTS_DIR, f'detailed_metrics_{config.KEY_TASK_FOR_DETAIL_CHART}.png')); plt.close()
        print(f"  3. Detailed metrics chart for '{config.KEY_TASK_FOR_DETAIL_CHART}' saved.")

    # VISUAL 4: Comprehensive Side-by-Side Confusion Matrices
    for task in common_tasks:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8)); fig.suptitle(f'Comparative Confusion Matrices for: {task.upper()}', fontsize=20)
        # Prelim
        if prelim_true.get(task):
            labels = np.unique(np.concatenate((prelim_true[task], prelim_pred[task]))); names = prelim_encoders[task].inverse_transform(labels)
            cm = confusion_matrix(prelim_true[task], prelim_pred[task], labels=labels)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=names.tolist(), yticklabels=names.tolist(), ax=axes[0]); axes[0].set_title('Preliminary Model (n=504)', fontsize=16)
        else:
            axes[0].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=20); axes[0].set_title('Preliminary Model (n=504)', fontsize=16)
        axes[0].set_ylabel('True Label'); axes[0].set_xlabel('Predicted Label')
        # Final
        if final_true.get(task):
            labels = np.unique(np.concatenate((final_true[task], final_pred[task]))); names = final_encoders[task].inverse_transform(labels)
            cm = confusion_matrix(final_true[task], final_pred[task], labels=labels)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=names.tolist(), yticklabels=names.tolist(), ax=axes[1]); axes[1].set_title('Final Model (n=9350)', fontsize=16)
        else:
            axes[1].text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=20); axes[1].set_title('Final Model (n=9350)', fontsize=16)
        axes[1].set_ylabel('True Label'); axes[1].set_xlabel('Predicted Label')
        plt.tight_layout(rect=(0, 0, 1, 0.96)); plt.savefig(os.path.join(config.RESULTS_DIR, f'4_comparative_cm_{task}.png')); plt.close()
    print("  4. All comparative confusion matrices saved.")

    # VISUAL 5: Per-Class F1-Score Heatmap
    heatmap_df = pd.DataFrame(per_class_metrics).pivot_table(index=['Task', 'Class'], columns='Model', values='f1-score')
    plt.figure(figsize=(12, 18)); sns.heatmap(heatmap_df.fillna(0), annot=True, cmap="viridis", fmt=".2f", linewidths=.5)
    plt.title('Heatmap of Per-Class F1-Scores', fontsize=18); plt.xlabel('Model Version'); plt.ylabel('Task and Class'); plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'per_class_f1_heatmap.png')); plt.close()
    print("  5. Per-class F1-score heatmap saved.")

    # VISUAL 6: Label Distribution of the Test Set
    fig, axes = plt.subplots(len(all_tasks), 1, figsize=(10, 5 * len(all_tasks)))
    fig.suptitle('Test Set Label Distribution', fontsize=20, y=1.0)
    for i, task in enumerate(all_tasks):
        if task in df_test_filtered.columns:
            sns.countplot(ax=axes[i], y=df_test_filtered[task], palette='pastel', hue=df_test_filtered[task], legend=False, order=df_test_filtered[task].value_counts().index)
            axes[i].set_title(f'Distribution for: {task.upper()}'); axes[i].set_ylabel('')
    plt.tight_layout(); plt.savefig(os.path.join(config.RESULTS_DIR, 'label_distribution.png')); plt.close()
    print("  6. Label distribution chart saved.")

    # VISUAL 7: Feature Correlation Heatmap
    corr_df = pd.DataFrame()
    for col, le in final_encoders.items():
        if col in df_test_filtered.columns: corr_df[col] = le.transform(df_test_filtered[col])
    plt.figure(figsize=(12, 10)); sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Heatmap in Test Data', fontsize=18); plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'feature_correlation_heatmap.png')); plt.close()
    print("  7. Feature correlation heatmap saved.")

    # VISUAL 8: Performance vs. Text Length
    df_test_filtered['text_length'] = df_test_filtered['text'].str.len()
    df_test_filtered['length_bin'] = pd.cut(df_test_filtered['text_length'], bins=[0, 50, 100, 150, 200, 300, 500], labels=['0-50', '51-100', '101-150', '151-200', '201-300', '301+'])
    length_scores = []
    for task in config.FINAL_LABEL_COLUMNS:
        if final_true.get(task):
            task_df = pd.DataFrame({'true': final_true[task], 'pred': final_pred[task], 'bin': df_test_filtered['length_bin']})
            for bin_name in task_df['bin'].unique().dropna():
                bin_df = task_df[task_df['bin'] == bin_name]
                score = f1_score(bin_df['true'], bin_df['pred'], average='weighted', zero_division=0)
                length_scores.append({'Task': task, 'Length Bin': bin_name, 'F1-Score': score})
    length_df = pd.DataFrame(length_scores)
    plt.figure(figsize=(14, 8)); sns.lineplot(data=length_df, x='Length Bin', y='F1-Score', hue='Task', marker='o', palette='tab10')
    plt.title('Final Model Performance vs. Text Length', fontsize=18); plt.ylabel('Weighted F1-Score'); plt.xlabel('Text Length (Characters)'); plt.legend(bbox_to_anchor=(1.05, 1), loc=2); plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'performance_vs_length.png')); plt.close()
    print("  8. Performance vs. text length chart saved.")

    # VISUAL 9: Per-Class Accuracy Bar Chart for Key Task
    if config.KEY_TASK_FOR_ACCURACY_CHART in all_tasks:
        acc_scores = []
    task = config.KEY_TASK_FOR_ACCURACY_CHART
    
    # --- FIX: Add a small epsilon to the denominator to prevent division by zero ---
    epsilon = 1e-9
    
    if task in config.PRELIM_LABEL_COLUMNS and prelim_true.get(task):
        cm = confusion_matrix(prelim_true[task], prelim_pred[task])
        # The diagonal is the number of correct predictions for each class
        # The sum along the row is the total number of true samples for each class
        accuracies = cm.diagonal() / (cm.sum(axis=1) + epsilon)
        for i, acc in enumerate(accuracies):
            if i < len(prelim_encoders[task].classes_): # Safety check
                acc_scores.append({'Class': prelim_encoders[task].classes_[i], 'Model': 'Preliminary (n=504)', 'Accuracy': acc})
    
    if task in config.FINAL_LABEL_COLUMNS and final_true.get(task):
        cm = confusion_matrix(final_true[task], final_pred[task])
        accuracies = cm.diagonal() / (cm.sum(axis=1) + epsilon)
        for i, acc in enumerate(accuracies):
            if i < len(final_encoders[task].classes_): # Safety check
                acc_scores.append({'Class': final_encoders[task].classes_[i], 'Model': 'Final (n=9350)', 'Accuracy': acc})
    
    if acc_scores:
        acc_df = pd.DataFrame(acc_scores)
        plt.figure(figsize=(12, 7)); sns.barplot(data=acc_df, x='Class', y='Accuracy', hue='Model', palette={'Preliminary (n=504)': 'salmon', 'Final (n=9350)': 'skyblue'})
        plt.title(f'Per-Class Accuracy (Recall) for Task: {task.upper()}', fontsize=18); plt.ylim(0, 1.05); plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_DIR, f'per_class_accuracy_{task}.png')); plt.close()
        print(f"  9. Per-class accuracy chart for '{task}' saved.")
            
    # VISUAL 10: Top Misclassifications Analysis
    print("\n--- Top Misclassifications for Final Model ---")
    for task in config.FINAL_LABEL_COLUMNS:
        if final_true.get(task) and len(final_encoders[task].classes_) > 2:
            cm = confusion_matrix(final_true[task], final_pred[task])
            np.fill_diagonal(cm, 0)
            if cm.max() > 0:
                max_idx = np.unravel_index(cm.argmax(), cm.shape)
                true_class = final_encoders[task].classes_[max_idx[0]]; pred_class = final_encoders[task].classes_[max_idx[1]]
                print(f"  - Task '{task}': Most often confused '{true_class}' with '{pred_class}' ({cm.max()} times).")
    
    # VISUAL 11: Model Speed Comparison
    speed_data = [{'Model': 'Preliminary (n=504)', 'Speed (samples/sec)': prelim_speed}, {'Model': 'Final (n=9350)', 'Speed (samples/sec)': final_speed}]
    speed_df = pd.DataFrame(speed_data)
    plt.figure(figsize=(8, 5)); sns.barplot(data=speed_df, x='Model', y='Speed (samples/sec)', hue='Model', palette={'Preliminary (n=504)': 'salmon', 'Final (n=9350)': 'skyblue'}, legend=False)
    plt.title('Model Inference Speed Comparison', fontsize=16); plt.ylabel('Samples per Second'); plt.xlabel('')
    for index, row in speed_df.iterrows(): plt.text(row.name, row['Speed (samples/sec)'], f"{row['Speed (samples/sec)']:.1f}", color='black', ha="center")
    plt.tight_layout(); plt.savefig(os.path.join(config.RESULTS_DIR, 'model_speed_comparison.png')); plt.close()
    print("  11. Model speed comparison chart saved.")

if __name__ == "__main__":
    main()