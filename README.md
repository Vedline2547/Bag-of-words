# Spam Message Classification Using Bag of Words and Naive Bayes

## Project Overview

This project implements a machine learning model that classifies SMS messages as either **Spam** or **Ham (Legitimate)** using Natural Language Processing (NLP) techniques. It uses the Bag of Words approach to convert text messages into numerical features and a Multinomial Naive Bayes classifier to predict whether a message is spam.

The project demonstrates a fundamental NLP workflow, from text preprocessing and feature extraction to model training, evaluation, and prediction on new messages.

## Objectives

- Explore and prepare an SMS spam dataset using Pandas.
- Convert categorical labels into numerical values.
- Transform text messages into numerical representations using `CountVectorizer`.
- Train a Multinomial Naive Bayes classification model.
- Evaluate model performance using precision, recall, F1-score, and support.
- Test the trained model on new text messages.

## Technologies and Libraries

- **Python** — Programming language
- **Pandas** — Data loading and manipulation
- **NumPy** — Numerical operations
- **Scikit-learn** — Data splitting, text vectorization, model training, and evaluation

## Machine Learning Workflow

### 1. Data Preparation

The SMS dataset is loaded from a CSV file containing message text and category labels. The original categories are converted into binary labels:

- `1` — Spam
- `0` — Ham (legitimate message)

### 2. Train-Test Split

The dataset is divided into training and testing sets using an 80:20 split. The training data is used to fit the model, while the testing data evaluates its performance on unseen messages.

### 3. Feature Extraction with Bag of Words

Scikit-learn's `CountVectorizer` converts text messages into numerical feature vectors based on word occurrence counts.

The vectorizer learns its vocabulary from the training messages and applies the same vocabulary to the testing messages.

### 4. Model Training

A `MultinomialNB` classifier is trained on the Bag of Words features. Naive Bayes estimates the likelihood of a message belonging to each class based on the words it contains.

### 5. Model Evaluation

The model is evaluated using Scikit-learn's `classification_report`, which provides:

- Precision
- Recall
- F1-score
- Support

These metrics help assess how effectively the model identifies spam messages while avoiding incorrect classifications of legitimate messages.

### 6. Prediction on New Messages

The trained vectorizer transforms new messages into numerical features, which are passed to the trained classifier to predict whether the message is spam or legitimate.

## Model and Techniques

| Component | Technique |
|---|---|
| Text representation | Bag of Words |
| Feature extraction | `CountVectorizer` |
| Classification algorithm | Multinomial Naive Bayes |
| Data splitting | Train-test split |
| Evaluation | Classification report |

## Example Prediction

```python
message = ["Click here to win 50000$"]

message_features = v.transform(message)
prediction = model.predict(message_features)

if prediction[0] == 1:
    print("Spam message")
else:
    print("Legitimate message (Ham)")
```

The model returns a binary prediction based on patterns learned from the training dataset.

## Key Learnings

- Understanding text classification as a supervised machine learning problem.
- Applying Bag of Words to convert unstructured text into numerical features.
- Understanding how Multinomial Naive Bayes can be used for text classification.
- Using sparse matrices and vocabulary indices to inspect text features.
- Evaluating classification models using multiple performance metrics.
- Applying a trained NLP pipeline to new text inputs.

## Limitations

- Bag of Words does not capture word order or semantic meaning.
- Model performance depends on the quality and representativeness of the training dataset.
- Words absent from the learned vocabulary do not contribute to the feature vector.
- Further improvements could include TF-IDF features, text preprocessing, hyperparameter tuning, and comparison with other classifiers.

## Future Improvements

- Compare Bag of Words with TF-IDF vectorization.
- Experiment with different classification algorithms.
- Tune model parameters using cross-validation.
- Analyze the confusion matrix and class-specific performance.
- Build an interactive application for testing messages.

## Conclusion

This project demonstrates how traditional NLP techniques and probabilistic machine learning can be combined to classify SMS messages. It provides a practical foundation for exploring text vectorization, supervised learning, and spam detection.

## Author

**Vedline Ochieng**

Civil Engineering Student | Machine Learning Enthusiast | Python Developer

- GitHub: [@Vedline2547](https://github.com/Vedline2547)
- LinkedIn: [Vedline Ochieng](https://www.linkedin.com/in/vedline-ochieng-025146290/)
