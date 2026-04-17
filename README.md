# NLP Writing Style Classification

This project analyzes the evolution of writing style in Charles Dickens’ novels using Natural Language Processing and machine learning techniques.

## Overview

The goal of this project is to classify texts into different writing periods (early, middle, late) based on stylometric features. The analysis is performed at the chapter level, allowing for a more granular understanding of stylistic changes over time.

## Approach

We extract a variety of linguistic and statistical features from the text, including:
- Part-of-speech (POS) distribution
- Word and lemma frequency
- Sentence length
- Sentiment analysis
- Verb tense usage
- Named entities
- Punctuation patterns

These features are used to train and evaluate multiple machine learning models.

## Models Used

- Logistic Regression
- Multi-Layer Perceptron (MLP)
- Decision Tree

Among these, Logistic Regression achieved the best performance (~78% accuracy).

## Key Findings

- Part-of-speech (POS) distribution was the most impactful feature for classification
- Punctuation usage (e.g., question and exclamation marks) also contributed significantly
- Some features (e.g., frequent lemmas) introduced noise and reduced model performance

## Tech Stack

- Python
- spaCy (NLP processing)
- scikit-learn (ML models)
- NLTK (sentiment analysis)
- Gensim (topic modeling)
- PyLDAvis (visualization)
- Pandas, NumPy (data processing)
- Matplotlib (visualization)

## Project Structure

- `main.py` – visualization and main execution
- `data_parsing.py` – data preprocessing and parsing
- `classifiers.py` – model training and evaluation
- `deviding_to_chaps.py` – splitting books into chapters

## Dataset

The dataset consists of Charles Dickens' novels, divided into chapters and grouped into three chronological writing periods.

## Notes

This project was developed as part of a team. My contributions included working on the NLP pipeline, feature extraction, and model evaluation.

## Full Report

A detailed analysis of the project, including methodology, experiments, and results, is available in:
[docs/project_report.pdf](docs/project_report.pdf)