# Slang-Aware Fake Review Detection

This project detects fake product reviews, especially in Tanglish (`Tamil + English`), using a hybrid deep learning and feature-based approach.

## Live Website

[Click here to use the live website](https://fake-review-detector-1--vishalatchi814.replit.app)

## Features

- Detects fake and genuine product reviews
- Focuses on Tanglish (`Tamil + English`) review text
- Uses MuRIL transformer for multilingual understanding
- Detects slang, over-exaggeration, and promotional wording
- Hybrid model combining transformer embeddings and behavioral features
- Flask-based web interface for real-time prediction

## Tech Stack

- Python
- Flask
- Transformers (`MuRIL`)
- Scikit-learn
- NumPy
- PyTorch

## How It Works

The system analyzes the input review using:
- MuRIL transformer embeddings
- handcrafted linguistic features
- exaggeration and review-behavior cues

It then predicts whether the review is:
- `Fake`
- `Genuine`

## How to Run Locally

```bash
pip install -r requirements.txt
python app.py

Then open:
http://127.0.0.1:5000

Project Objective
The goal of this project is to improve fake review detection for code-mixed Tanglish reviews, where traditional text models like TF-IDF were less effective. This project uses a smarter hybrid approach to better identify exaggerated, misleading, and unnatural review patterns.

Note
Due to GitHub file size limitations, trained model files may not be uploaded completely in this repository.
The full system works correctly in the local environment with trained model files, and the deployed live version is available at the link above.
