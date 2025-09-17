# Document Classification and Information Extraction

## Project Overview

This project implements a document classification and information extraction system using traditional machine learning techniques. The system can classify documents into different categories and extract structured information from invoices.

### Objectives

- **Classify documents** into distinct categories
- **Extract specific information** from invoices using traditional AI techniques
- Implement solution without using generative AI models

## Our Solution

### Document Categories

We classify documents into four categories:

- **Invoices**: Financial documents requesting payment
- **Receipts**: Proof of completed transactions
- **Emails**: Electronic communication documents
- **Contracts**: Legal agreement documents
- **Others**: anything possible

### Classification Approach

Our document classification pipeline:

1. Data preprocessing and tokenization
2. Feature extraction
3. Train/test split for model validation
4. Model training and evaluation
5. Hyperparameter tuning

### Information Extraction

For documents classified as invoices, we extract:

- Invoice number
- Invoice date
- Due date
- Issuer name
- Recipient name
- Total amount

## Current Results

```
Classification Report:
              precision    recall  f1-score   support

           1       1.00      0.95      0.97        60
           2       0.92      0.98      0.95        60
           3       0.90      0.93      0.92        60
           4       0.98      0.95      0.97        60
           5       0.98      0.97      0.97        60

    accuracy                           0.96       300
   macro avg       0.96      0.96      0.96       300
weighted avg       0.96      0.96      0.96       300
```

## Setup and Usage

### Requirements

- Python 3.8+
- Required libraries (see requirements.txt)

### Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download and prepare the dataset

### Running the System

1. go i UI repo: `cd ui`
2. For document classification: `python app.py`

## Project Requirements

### 1. Document Classification

- Design and implement a system that classifies documents into **at least four (4) distinct categories**
- One category **must be invoices**
- Provide **justification** for selected categories and describe **key characteristics**

### 2. Information Extraction

- **Automatically extract structured information** from documents classified as **invoices**
- Extract: invoice number, dates, issuer/recipient names, and total amount

### Technical Constraints

- No generative AI allowed
- Use traditional techniques:
  - Text classification methods
  - Pattern analysis
  - Rule-based approaches with regular expressions
  - Classical Machine Learning models
  - Traditional NLP and/or CV methods

## Evaluation Criteria

- **Accuracy** in document classification
- **Precision and completeness** in information extraction

