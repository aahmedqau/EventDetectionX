Title: RLLaMA-BERT: A Hybrid Transformer Framework for Event Detection on X (Twitter)

Description: This repository implements the hybrid RankRAG-BERT model proposed in the paper “A Framework for Event Detection on X Based on Transformer Model.”
The system combines a ranking-based large language model (RankRAG / RLLaMA-3) for context enrichment with BERT for event classification. It detects event-related tweets in real time by filtering and classifying large social media datasets.

Dataset Information

Two datasets were used:

US Elections Dataset – Political event tweets annotated as event/non-event.

FA Cup Dataset – Sports-related tweets capturing match events (e.g., goals, saves).

Both datasets include:

Tweet text

Timestamp

Hashtags

Contextual metadata (user mentions, URLs, etc.)


Code Information:
rankrag_topk_selection.py	(Fine-tunes RankRAG (RLLaMA-3) to select top-K relevant tweets.)
event_detection_bert.py		(Classifies tweets using BERT-based transformer.)
evaluation_metrics.py		(Computes precision, recall, F1-score, and accuracy.)
train.py			(Coordinates pipeline execution, training, and evaluation)

Usage Instructions:

Clone Repository
git clone https://github.com/yourusername/rllama-bert-event-detection.git
cd rllama-bert-event-detection


Install Dependencies
pip install torch transformers datasets scikit-learn pandas numpy


Run RankRAG Tweet Selection
python rankrag_topk_selection.py --input data/US_Elections.csv --k 2000

Run Event Detection using BERT
python event_detection_bert.py --input outputs/topk_tweets.json --model bert-base-uncased




