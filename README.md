Title: RLLaMA-BERT: A Hybrid Transformer Framework for Event Detection on X (Twitter)

Description: This repository implements the hybrid RankRAG-BERT model proposed in the paper “A Framework for Event Detection on X Based on Transformer Model.”
The system combines a ranking-based large language model (RankRAG / RLLaMA-3) for context enrichment with BERT for event classification. It detects event-related tweets in real time by filtering and classifying large social media datasets.

Dataset Information

Two datasets were used:

USElectionDataset – Political event instances

FACupDataset – Sports-related event instances

Code Information:
Run Event Detection.py

Usage Instructions
1.	We have the required CSV files in the correct format
2.	All dependencies are installed (transformers, sentence-transformers, faiss-cpu, etc.)
3.	We follow the execution flow as written in the Python script
4.	Ensure CSV files (train.csv, valid.csv, test.csv) follow the exact column format.
5.	The retrieval system requires either a corpus.csv or will fall back to using training tweets
6.	Each event instance gets augmented with top-k retrieved contexts before training

For baseline comparison you can run Python scripts: Comparison with baselines using FA Cup Dataset.py,  Comparison with baselines using US Election Dataset.py





