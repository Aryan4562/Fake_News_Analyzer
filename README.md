Project Overview:

The Fake News Analyzer is a web-based application that detects whether a given news text is Real or Fake using Natural Language Processing (NLP) and Machine Learning techniques.

The system uses a Flask backend for model inference and a React (Vite) frontend for user interaction. The ML model is trained on a labeled news dataset and uses text preprocessing and feature extraction techniques to classify the input text.

This project demonstrates the practical application of NLP, Machine Learning, and Full-Stack Development in solving real-world problems like misinformation detection.

🎯 Objectives

To analyze news text and classify it as Fake or Real

To apply NLP preprocessing techniques such as tokenization, stopword removal, and lemmatization

To build a machine learning-based text classification model

To provide a user-friendly web interface for testing news articles

To demonstrate end-to-end integration of ML + Backend + Frontend

🛠️ Tech Stack
🔹 Frontend

React (Vite)

JavaScript

HTML & CSS

🔹 Backend

Python

Flask (REST API)

🔹 Machine Learning / NLP

NLTK

Scikit-learn

Pandas, NumPy

TF-IDF / Count Vectorizer (for feature extraction)

Machine Learning Model (e.g., Random Forest / Logistic Regression)

⚙️ System Architecture

User enters news text in the web interface

React frontend sends the text to Flask API

Flask backend:

Preprocesses the text using NLP techniques

Converts text into numerical features

Passes features to the trained ML model

Model predicts Fake or Real

Result is sent back and displayed on the frontend

✨ Features

Clean and simple user interface

Real-time news text analysis

NLP-based text preprocessing

Machine Learning-based classification

Displays prediction result (Fake / Real)

Easy to extend with more models or datasets