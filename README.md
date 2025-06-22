# 🧠 NLP Transformers: Fine-Tuning, Commonsense & Sentence Similarity

## Description

This project explores the fine-tuning and application of transformer-based language models on two core NLP tasks:

1. **Sentiment Classification** – Using the [Yelp Polarity](https://huggingface.co/datasets/yelp_polarity) dataset to train and evaluate binary sentiment classifiers.
2. **Commonsense Reasoning** – Using the [Winogrande](https://huggingface.co/datasets/winogrande) and [PIQA](https://huggingface.co/datasets/piqa) datasets to reformulate reasoning problems as:
   - **Binary classification**
   - **Natural Language Inference (NLI)**
   - **Semantic similarity** with sentence embeddings

All experiments are implemented in a single Jupyter notebook using the Hugging Face ecosystem (`transformers`, `datasets`, `Trainer`, and `pipelines`).

---

## Setup

Install all required packages:

```bash
pip install torch transformers datasets sentence-transformers scikit-learn numpy pandas
```

**Tested with:**
- Python 3.10
- PyTorch 2.2
- Transformers 4.40
- Datasets 2.19
- Sentence-Transformers 2.7
- scikit-learn 1.5

---

## Task 1 – Sentiment Classification (Yelp)

**Goal:** Fine-tune transformer models to classify restaurant reviews as either positive or negative.

### Steps:
- Load and preprocess the Yelp Polarity dataset
- Tokenize using `AutoTokenizer` (e.g., BERT, RoBERTa)
- Fine-tune using Hugging Face’s `Trainer` API
- Evaluate using accuracy and confusion matrix
- Experiment with different hyperparameters:
  - Learning rate
  - Batch size
  - Number of epochs
  - Weight decay
- Optional: Compare full fine-tuning vs. sentence embeddings + logistic regression

---

## Task 2 – Commonsense Reasoning (Winogrande & PIQA)

**Goal:** Use pretrained models to solve fill-in-the-blank problems that require commonsense knowledge.

### A. Binary Classification Approach
- Replace the blank with each candidate
- Classify which sentence is more likely to be valid using a binary classifier

### B. NLI Reformulation
- Treat original sentence as **premise**, and candidate-filled sentences as **hypotheses**
- Use an NLI model to predict **entailment** vs **contradiction**
- Select the hypothesis with the highest entailment probability

### C. Semantic Similarity with Sentence Transformers
- Generate sentence embeddings using models like `sentence-t5-base`
- Use cosine similarity to score candidate answers against a reference (e.g., “best answer”)
- Evaluate alignment between semantic similarity and classification accuracy

---

## Evaluation Summary

- Binary and NLI approaches were compared across models (`roberta-large`, `bart-large-mnli`, `deberta-v3-base`)
- Sentence similarity scores from models like `all-mpnet-base-v2` and `sentence-t5-base` correlated with classification outcomes
- Semantic similarity proved helpful in evaluating generated or alternative answers when gold labels were ambiguous

---

## Educational Context

This notebook was developed as part of the course **"Neural Networks and Deep Learning"**  
📍 School of Electrical and Computer Engineering, NTUA  
The objective was to explore real-world applications of pretrained models through fine-tuning and zero-shot reasoning.

---

## Author

**Nikolaos Katsaidonis**  
Electrical & Computer Engineering, NTUA

