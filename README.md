# Language_Models_Transfer_Learning_and_Hyperparameters

## Description

This project contains a deep learning notebook that explores two tasks using transformer models:

1. **Sentiment classification** using the Yelp Polarity dataset.
2. **Commonsense reasoning** using the Winogrande dataset with two modeling strategies: binary classification and Natural Language Inference (NLI).

All code is implemented in a single Jupyter notebook using Hugging Face Transformers and Datasets libraries.

---

## Requirements

Install all required libraries with:

```bash
pip install torch transformers datasets sentence-transformers scikit-learn numpy pandas
```

Tested with:
- Python 3.10
- PyTorch 2.2
- Transformers 4.40
- Datasets 2.19
- Sentence-Transformers 2.7
- scikit-learn 1.5

---


## Task 1 – Yelp Polarity Sentiment Classification

**Objective:** Train a transformer-based model to classify reviews as positive or negative.

### Workflow:
- Load the Yelp Polarity dataset using Hugging Face `datasets`.
- Explore data: review text and polarity labels (0: negative, 1: positive).
- Tokenize text using `AutoTokenizer` with max_length 128.
- Use models such as `bert-base-uncased`, `roberta-base`, etc.
- Fine-tune using the Hugging Face `Trainer` API:
  - Adjust hyperparameters: learning rate, batch size, epochs
  - Evaluate accuracy on test set using `compute_metrics` function
- Visualize training and validation loss
- Perform hyperparameter tuning (learning rate, batch size, epochs)
- Compare results across different models


## Task 2 – Winogrande Commonsense Reasoning

**Objective:** Predict the correct completion in a sentence that requires common sense knowledge.

### A. Binary Classification Approach
- Replace blank with each choice
- Format as `[CLS] sentence_with_choice [SEP]`
- Fine-tune `roberta-large` as binary classifier
- Predict based on which sentence yields higher probability

### B. Natural Language Inference (NLI) Approach
- Treat the original sentence as a premise
- Fill the blank with each choice to create two hypotheses
- Label: entailment for correct, contradiction for incorrect
- Train a model (e.g., `deberta-v3-large`) to classify entailment
- Choose the hypothesis with highest entailment probability

---


The project was created as part of the course "Neural Networks and Deep Learning" at School of Electrical and Computer Engineering, NTUA and the aim of this work is to present my approach to the problems given. 



## Author

**Nikolaos Katsaidonis**  
