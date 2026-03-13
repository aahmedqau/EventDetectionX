import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import transformers
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
from transformers import XLNetTokenizer, XLNetModel, DistilBertTokenizer, DistilBertModel
import warnings
import random
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================================
# DATASET PREPARATION
# ==========================================================
class ElectionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TransformerDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# ==========================================================
# DATA LOADING AND PREPROCESSING
# ==========================================================
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    # Create features WITHOUT leaking the target
    df['vote_share'] = df['candidatevotes'] / df['totalvotes'] * 100
    df['log_votes'] = np.log1p(df['candidatevotes'])
    df['log_total'] = np.log1p(df['totalvotes'])
    df['vote_ratio'] = df['candidatevotes'] / (df['totalvotes'] + 1)
    
    # Encode categorical variables
    le_party = LabelEncoder()
    le_state = LabelEncoder()
    
    df['party_encoded'] = le_party.fit_transform(df['party_simplified'].fillna('OTHER'))
    df['state_encoded'] = le_state.fit_transform(df['state'])
    
    # Features - EXCLUDE party_simplified to prevent leakage
    feature_cols = ['year', 'state_encoded', 'vote_share', 'log_votes', 
                   'log_total', 'vote_ratio', 'totalvotes']
    X = df[feature_cols].values
    
    # Target: Predict if candidate is Democrat (more realistic task)
    y = (df['party_simplified'].isin(['DEMOCRAT'])).astype(int).values
    
    return X, y, df

# ==========================================================
# BASE MODELS (with increased regularization)
# ==========================================================
class BERTClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.5)  # Increased dropout
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)

class RoBERTaClassifier(nn.Module):
    def __init__(self, n_classes):
        super(RoBERTaClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.drop = nn.Dropout(p=0.5)
        self.out = nn.Linear(self.roberta.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)

class XLNetClassifier(nn.Module):
    def __init__(self, n_classes):
        super(XLNetClassifier, self).__init__()
        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
        self.drop = nn.Dropout(p=0.5)
        self.out = nn.Linear(self.xlnet.config.d_model, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.xlnet(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(outputs[0][:, -1, :])
        return self.out(output)

class DistilBERTClassifier(nn.Module):
    def __init__(self, n_classes):
        super(DistilBERTClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.drop = nn.Dropout(p=0.5)
        self.out = nn.Linear(self.distilbert.config.dim, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0]
        output = self.drop(pooled_output)
        return self.out(output)

class RLLaMABERTClassifier(nn.Module):
    def __init__(self, n_classes):
        super(RLLaMABERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.5)  # Increased from 0.4
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, n_classes)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        output = self.fc1(output)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.drop(output)
        output = self.fc2(output)
        output = self.bn2(output)
        output = self.relu(output)
        return self.out(output)

# ==========================================================
# TRAINING FUNCTION (with early stopping & noise injection)
# ==========================================================
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=3):  # Reduced epochs
    best_val_acc = 0
    best_model_state = None
    patience = 2
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training with noise injection
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Add label smoothing noise (5% chance to flip label)
            if random.random() < 0.05:
                labels = 1 - labels
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model

# ==========================================================
# EVALUATION FUNCTION
# ==========================================================
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return accuracy, precision, recall, f1

# ==========================================================
# TEXT PREPROCESSING FOR TRANSFORMERS (FIXED - NO LEAKAGE)
# ==========================================================
def create_text_features(df):
    """Create text descriptions WITHOUT party name to prevent data leakage"""
    texts = []
    for _, row in df.iterrows():
        # REMOVED party_simplified and candidate name to prevent trivial classification
        text = f"Year {row['year']} {row['state']} {row['office']} votes {row['candidatevotes']} total {row['totalvotes']}"
        texts.append(text)
    return texts

# ==========================================================
# MAIN EXPERIMENT
# ==========================================================
def main():
    print("Loading and preprocessing data...")
    X, y, df = load_and_preprocess_data('/content/USElectionDataset.csv')

    # Create text features (without party leakage)
    texts = create_text_features(df)

    # Split data with larger validation set
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, y, test_size=0.25, random_state=42, stratify=y  # Increased test size
    )

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.15, random_state=42, stratify=train_labels  # Larger val set
    )

    # Define models to compare
    models_config = {
        'RLLaMA-BERT': (RLLaMABERTClassifier, 'bert-base-uncased', 2),
        'BERT': (BERTClassifier, 'bert-base-uncased', 2),
        'RoBERTa': (RoBERTaClassifier, 'roberta-base', 2),
        'XLNet': (XLNetClassifier, 'xlnet-base-cased', 2),
        'DistilBERT': (DistilBERTClassifier, 'distilbert-base-uncased', 2)
    }

    results = []

    for model_name, (model_class, pretrained_name, n_classes) in models_config.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name}...")
        print(f"{'='*60}")

        # Tokenize
        if 'roberta' in pretrained_name:
            tokenizer = RobertaTokenizer.from_pretrained(pretrained_name)
        elif 'xlnet' in pretrained_name:
            tokenizer = XLNetTokenizer.from_pretrained(pretrained_name)
        elif 'distilbert' in pretrained_name:
            tokenizer = DistilBertTokenizer.from_pretrained(pretrained_name)
        else:
            tokenizer = BertTokenizer.from_pretrained(pretrained_name)

        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
        test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

        # Create datasets
        train_dataset = TransformerDataset(train_encodings, train_labels)
        val_dataset = TransformerDataset(val_encodings, val_labels)
        test_dataset = TransformerDataset(test_encodings, test_labels)

        # Create dataloaders with smaller batch size for better generalization
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Reduced batch size
        val_loader = DataLoader(val_dataset, batch_size=16)
        test_loader = DataLoader(test_dataset, batch_size=16)

        # Initialize model
        model = model_class(n_classes).to(device)

        # Loss and optimizer with weight decay for regularization
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)  # Lower LR + weight decay

        # Train with fewer epochs
        model = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=3)

        # Evaluate
        accuracy, precision, recall, f1 = evaluate_model(model, test_loader)

        results.append({
            'Model': model_name,
            'Accuracy': f"{accuracy*100:.1f}%",
            'Precision': f"{precision:.2f}",
            'Recall': f"{recall:.2f}",
            'F1-Score': f"{f1:.2f}"
        })

        print(f"{model_name} Results:")
        print(f"  Accuracy:  {accuracy*100:.1f}%")
        print(f"  Precision: {precision:.2f}")
        print(f"  Recall:    {recall:.2f}")
        print(f"  F1-Score:  {f1:.2f}")

    # Display results table
    print("\n" + "="*80)
    print("Table: Comparison of Proposed Model with Existing Approaches on US Election Dataset")
    print("="*80)
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*80)
    for result in results:
        print(f"{result['Model']:<20} {result['Accuracy']:<12} {result['Precision']:<12} {result['Recall']:<12} {result['F1-Score']:<12}")
    print("="*80)

if __name__ == "__main__":
    main()