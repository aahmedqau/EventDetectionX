"""
RankRAG + BERT Event Detection (complete, runnable)

Python 3.9+
pip install -U transformers datasets sentence-transformers faiss-cpu scikit-learn torch accelerate

Expected files (optional):
- train.csv, valid.csv, test.csv  -> columns: tweet,label   (label is int or str)
- corpus.csv (optional)           -> column: text           (external retrieval corpus)

If files are missing, a small synthetic dataset is created automatically.

Author: (you)
"""

import os
import math
import random
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
from sklearn.preprocessing import LabelEncoder

import torch
from torch import nn

from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding, set_seed
)

# Bi-encoder for retrieval & Cross-Encoder for reranking
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import faiss


# --------------------------
# Configuration
# --------------------------
@dataclass
class Config:
    seed: int = 42
    top_k_retrieve: int = 20   # initial retrieve
    top_k_final: int = 5       # after rerank (diagram's "Top-k")
    max_length: int = 256
    bert_model: str = "bert-base-uncased"
    biencoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    crossencoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    train_path: str = "train.csv"
    valid_path: str = "valid.csv"
    test_path: str  = "test.csv"
    corpus_path: str = "corpus.csv"  # optional
    output_dir: str = "./rankrag_bert_event_model"
    num_train_epochs: int = 2
    per_device_batch_size: int = 16
    lr: float = 2e-5
    warmup_ratio: float = 0.1
    fp16: bool = True


CFG = Config()
set_seed(CFG.seed)


# --------------------------
# Utility: make toy data if none
# --------------------------
def maybe_make_toy_data():
    if all(os.path.exists(p) for p in [CFG.train_path, CFG.valid_path, CFG.test_path]):
        return
    print("[Info] Dataset CSVs not found. Creating a tiny toy dataset...")
    import pandas as pd
    events = [
        ("Earthquake felt in city, buildings shaking", "disaster"),
        ("Breaking: flood warning issued near river banks", "disaster"),
        ("Concert tonight at central park featuring local bands", "non_event"),
        ("Football finals match postponed due to rain", "event"),
        ("Wildfire spreading towards residential area, evacuations start", "disaster"),
        ("Job fair announced for graduates this weekend", "event"),
        ("Coffee shop opens new branch downtown", "non_event"),
        ("Tornado sirens reported, people seek shelter", "disaster"),
        ("Science expo kicks off with robotics showcase", "event"),
        ("Local bakery introduces seasonal pastries", "non_event"),
    ]
    random.shuffle(events)
    df = pd.DataFrame(events, columns=["tweet", "label"])
    train = df.sample(frac=0.7, random_state=CFG.seed)
    tmp = df.drop(train.index)
    valid = tmp.sample(frac=0.5, random_state=CFG.seed)
    test = tmp.drop(valid.index)
    train.to_csv(CFG.train_path, index=False)
    valid.to_csv(CFG.valid_path, index=False)
    test.to_csv(CFG.test_path, index=False)

    # simple retrieval corpus
    corpus_texts = [
        "An earthquake is a sudden shaking of the ground.",
        "Flood warnings are issued when rivers overflow the banks.",
        "Concerts attract large crowds for live music.",
        "Tornado sirens indicate severe weather conditions.",
        "Wildfires are uncontrolled fires in natural areas.",
        "Sports events like football finals engage many fans.",
        "Job fairs are recruiting events for employers and graduates.",
        "Bakeries sell pastries and breads.",
        "Science expos display technology and robotics."
    ]
    pd.DataFrame({"text": corpus_texts}).to_csv(CFG.corpus_path, index=False)


maybe_make_toy_data()


# --------------------------
# Load labeled data
# --------------------------
import pandas as pd

train_df = pd.read_csv(CFG.train_path)
valid_df = pd.read_csv(CFG.valid_path)
test_df  = pd.read_csv(CFG.test_path)

# Normalize column names
train_df.columns = [c.strip().lower() for c in train_df.columns]
valid_df.columns = [c.strip().lower() for c in valid_df.columns]
test_df.columns  = [c.strip().lower() for c in test_df.columns]

assert "tweet" in train_df.columns and "label" in train_df.columns, "CSV must have columns: tweet,label"

# Encode labels
label_encoder = LabelEncoder()
all_labels = pd.concat([train_df["label"], valid_df["label"], test_df["label"]]).astype(str)
label_encoder.fit(all_labels)
num_labels = len(label_encoder.classes_)
train_df["label_id"] = label_encoder.transform(train_df["label"].astype(str))
valid_df["label_id"] = label_encoder.transform(valid_df["label"].astype(str))
test_df["label_id"]  = label_encoder.transform(test_df["label"].astype(str))

print(f"[Info] Labels: {list(label_encoder.classes_)}")


# --------------------------
# RankRAG: Context Retrieval + Reranking
# --------------------------
class RankRAG:
    """
    Implements:
      1) 'Instruction tuning' heuristic scorer (LLM-like prior) -> here, a simple keyword prior.
      2) Retrieval using bi-encoder + FAISS
      3) Rerank with cross-encoder
      4) Select top-k and concatenate with tweet
    """
    def __init__(self,
                 biencoder_model: str,
                 crossencoder_model: str,
                 top_k_retrieve: int = 20,
                 top_k_final: int = 5,
                 device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.biencoder = SentenceTransformer(biencoder_model, device=self.device)
        self.crossencoder = CrossEncoder(crossencoder_model, max_length=256, device=self.device)
        self.top_k_retrieve = top_k_retrieve
        self.top_k_final = top_k_final

        # Load corpus for retrieval (external or fall back to training tweets)
        if os.path.exists(CFG.corpus_path):
            cdf = pd.read_csv(CFG.corpus_path)
            cdf.columns = [c.strip().lower() for c in cdf.columns]
            assert "text" in cdf.columns, "corpus.csv must have a 'text' column"
            self.corpus = cdf["text"].astype(str).tolist()
        else:
            self.corpus = pd.concat([train_df["tweet"], valid_df["tweet"], test_df["tweet"]]).astype(str).tolist()

        print(f"[RankRAG] Corpus size = {len(self.corpus)}")

        # Build FAISS index
        self.corpus_embeddings = self.biencoder.encode(self.corpus, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
        dim = self.corpus_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine with normalized vectors
        self.index.add(self.corpus_embeddings)

        # Precompute for instruction prior (keyword → prior score)
        self.keywords = {
            "earthquake": 1.0, "flood": 0.9, "wildfire": 0.9, "tornado": 0.9,
            "concert": 0.6, "job fair": 0.7, "football": 0.6, "expo": 0.7,
            "bakery": 0.2, "coffee": 0.2
        }

    def instruction_prior(self, query: str) -> float:
        """Lightweight 'instruction tuning' prior: boosts disaster/event-y queries."""
        q = query.lower()
        score = 0.0
        for k, w in self.keywords.items():
            if k in q:
                score = max(score, w)
        return score

    def retrieve(self, query: str, k: int) -> List[Tuple[int, float]]:
        q_emb = self.biencoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        sims, idxs = self.index.search(q_emb, k)
        return [(int(idxs[0][i]), float(sims[0][i])) for i in range(len(idxs[0]))]

    def rerank(self, query: str, candidates: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        texts = [self.corpus[i] for i, _ in candidates]
        pairs = [(query, t) for t in texts]
        scores = self.crossencoder.predict(pairs).tolist()
        # combine with initial similarity and instruction prior
        prior = self.instruction_prior(query)
        combined = []
        for (i, sim), s in zip(candidates, scores):
            # Weighted sum: cross-encoder dominates, with small sim and prior influence
            combined_score = 0.75 * s + 0.20 * sim + 0.05 * prior
            combined.append((i, combined_score))
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined

    def topk_context(self, query: str) -> List[str]:
        cand = self.retrieve(query, self.top_k_retrieve)
        reranked = self.rerank(query, cand)[: self.top_k_final]
        return [self.corpus[i] for i, _ in reranked]


rankrag = RankRAG(
    biencoder_model=CFG.biencoder_model,
    crossencoder_model=CFG.crossencoder_model,
    top_k_retrieve=CFG.top_k_retrieve,
    top_k_final=CFG.top_k_final
)


def build_augmented_text(tweet: str, contexts: List[str]) -> str:
    """
    Concatenate tweet + top-k contexts.
    Mimics 'Context Rich Fine Tuning' + 'Classify Top-k Tweets' in the diagram.
    """
    ctx = " ".join([f"[CTX{i+1}] {c}" for i, c in enumerate(contexts)])
    return f"[TWEET] {tweet} {ctx}"


# --------------------------
# Prepare HF datasets with augmented text
# --------------------------
def apply_rankrag_augmentation(df: pd.DataFrame) -> pd.DataFrame:
    texts = []
    for t in df["tweet"].astype(str).tolist():
        topk = rankrag.topk_context(t)
        aug = build_augmented_text(t, topk)
        texts.append(aug)
    out = df.copy()
    out["aug_text"] = texts
    return out

print("[Info] Building RankRAG-augmented texts (train/valid/test)...")
train_aug = apply_rankrag_augmentation(train_df)
valid_aug = apply_rankrag_augmentation(valid_df)
test_aug  = apply_rankrag_augmentation(test_df)

hf = DatasetDict({
    "train": Dataset.from_pandas(train_aug[["aug_text", "label_id"]]),
    "validation": Dataset.from_pandas(valid_aug[["aug_text", "label_id"]]),
    "test": Dataset.from_pandas(test_aug[["aug_text", "label_id"]]),
})


# --------------------------
# Tokenization (acts like Patch Embeddings + Norm in diagram)
# --------------------------
tokenizer = AutoTokenizer.from_pretrained(CFG.bert_model, use_fast=True)

def tokenize_batch(batch):
    return tokenizer(
        batch["aug_text"],
        truncation=True,
        max_length=CFG.max_length
    )

tokenized = hf.map(tokenize_batch, batched=True, remove_columns=["aug_text"])
tokenized = tokenized.rename_column("label_id", "labels")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# --------------------------
# Model (Transformer Encoder x12 + Linear head)
# --------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    CFG.bert_model,
    num_labels=num_labels
)

# --------------------------
# Metrics
# --------------------------
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0))
    }


# --------------------------
# Train
# --------------------------
os.makedirs(CFG.output_dir, exist_ok=True)

total_train_steps = (len(tokenized["train"]) // CFG.per_device_batch_size) * CFG.num_train_epochs
print(f"[Info] Training samples: {len(tokenized['train'])}, steps≈{total_train_steps}")

training_args = TrainingArguments(
    output_dir=CFG.output_dir,
    per_device_train_batch_size=CFG.per_device_batch_size,
    per_device_eval_batch_size=CFG.per_device_batch_size,
    num_train_epochs=CFG.num_train_epochs,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=CFG.lr,
    warmup_ratio=CFG.warmup_ratio,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    fp16=CFG.fp16,
    logging_steps=20,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

# --------------------------
# Evaluate on test
# --------------------------
metrics = trainer.evaluate(tokenized["test"])
print("[Test metrics]", metrics)

# Save label mapping for inference
with open(os.path.join(CFG.output_dir, "labels.json"), "w") as f:
    json.dump({int(i): cls for i, cls in enumerate(label_encoder.classes_)}, f, indent=2)


# --------------------------
# Inference utility
# --------------------------
class EventDetector:
    def __init__(self, cfg: Config, label_encoder: LabelEncoder, rankrag: RankRAG):
        self.cfg = cfg
        self.label_encoder = label_encoder
        self.rankrag = rankrag
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.output_dir if os.path.exists(cfg.output_dir) else cfg.bert_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(cfg.output_dir if os.path.exists(cfg.output_dir) else cfg.bert_model)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, tweet: str) -> Dict:
        topk = self.rankrag.topk_context(tweet)
        text = build_augmented_text(tweet, topk)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.cfg.max_length).to(self.device)
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_id = int(np.argmax(probs))
        label = self.label_encoder.inverse_transform([pred_id])[0]
        return {
            "tweet": tweet,
            "contexts": topk,
            "pred_label": label,
            "probs": {self.label_encoder.inverse_transform([i])[0]: float(p) for i, p in enumerate(probs)}
        }


# Quick demo after training (uses top-k contexts)
detector = EventDetector(CFG, label_encoder, rankrag)
demo_tweets = [
    "Strong tremors reported; people rushing out of buildings!",
    "Huge turnout expected for tonight's rock concert!",
    "Grand opening of a café downtown."
]
for tw in demo_tweets:
    out = detector.predict(tw)
    print("\n[TWEET]", tw)
    print("Pred:", out["pred_label"])
    print("Top-2 contexts:", out["contexts"][:2])
