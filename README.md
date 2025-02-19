Overview

This project implements Logistic Regression for sentiment analysis using sparse feature vectors. The model is trained using gradient descent with L2 regularization and generates a learning curve to evaluate performance.

Features


Sparse Feature Representation: Uses HashMap for efficient storage of word presence.

Gradient Descent Optimization: Updates weights iteratively to minimize classification error.

Learning Curve Analysis: Evaluates model performance as training data size increases.

Evaluation Metrics: Computes precision, recall, and F1-score for classification accuracy.

Data Import & Export: Reads and writes vectors and labels for easy dataset handling.

Project Structure

├── LogisticRegression/

│   ├── LearningCurveGenerator.java  # Generates learning curves

│   ├── LogisticRegression.java      # Implements Logistic Regression training & classification

│   ├── Run.java                     # Main entry point for training & evaluation

├── Utils/

│   ├── EvaluationMetrics.java       # Computes precision, recall, and F1-score

│   ├── VectorImporter.java          # Imports vectors and labels from files

How It Works

Train the Model: The classifier learns word weights from labeled text data.

Predict Sentiment: Given a new text sample, the model predicts whether it's positive or negative.

Evaluate Performance: Computes precision, recall, and F1-score to measure accuracy.

Generate Learning Curve: Analyzes how model accuracy improves with more training data.

Usage

Running the Project


javac LogisticRegression/Run.java
java LogisticRegression.Run


Input Data


Store training and test dataset vectors in .txt files.

Update file paths in Run.java.


Output Files


learning_curve_logistic.csv: Learning curve data.

train_vectors.txt, train_labels.txt: Training dataset.

test_vectors.txt, test_labels.txt: Testing dataset.



