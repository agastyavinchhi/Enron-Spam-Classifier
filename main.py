# Name: Agastya Deepak Vinchhi
# Netid: av351
# Project 2: Enron Spam Classification with Logistic Regression and Naive Bayes

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

ENRON_FILE_PATH = "enron_spam_data.csv"


def load_data(file_path, max_features=1000):
    df = pd.read_csv(file_path)
    message_list = df["Message"].to_list()
    df_good = df[pd.isna(message_list) == False]
    message_list_good = df_good["Message"].to_list()
    vectorizer = CountVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(message_list_good).toarray()
    y = np.array(df_good["Spam/Ham"] == "spam").astype(int)
    return X, y, vectorizer


def get_k_fold_indices(n_samples, k):
    fold_ids = np.arange(0, n_samples)
    np.random.shuffle(fold_ids)
    fold_ids = fold_ids % k
    return fold_ids


def train_test_split(X, y, k=5, test_fold=1):
    fold_ids = get_k_fold_indices(X.shape[0], k)
    X_train = X[fold_ids != test_fold, :]
    X_test = X[fold_ids == test_fold, :]
    y_train = y[fold_ids != test_fold]
    y_test = y[fold_ids == test_fold]
    return X_train, X_test, y_train, y_test


def add_bias_column(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


def predict_probability(X, weights):
    # predict probability using sigmoid 1 / (1 + e^(−z)) where z = wTx
    return 1 / (1 + np.exp(-(X @ weights)))


def compute_loss(X, y, weights):
    # let us keep probability in between 0 and 1, as log(0) is NaN
    p = np.clip(predict_probability(X, weights), 1e-10, 1 - 1e-10)
    # binary cross-entropy loss (log loss) -1/n Σ(y*log(y^)+(1-y)*log(1-y^))
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def gradient_descent_step(X, y, weights, learning_rate, regularization="l1", lam=0.01):
    # gradient descent equation (1/n) Σ (y^ − y)x
    gradient = X.T @ (predict_probability(X, weights) - y) / X.shape[0]
    # l2: gradient += lambda*w
    if regularization == "l2":
        gradient[1:] += lam * weights[1:]
    # l1: gradient += lambda*np.sign(w)
    elif regularization == "l1":
        gradient[1:] += lam * np.sign(weights[1:])
    return weights - learning_rate * gradient


def train_logistic_regression(
    X,
    y,
    learning_rate=0.3,
    regularization="l1",
    lam=0.01,
    max_iterations=3000,
    tolerance=1e-4,
):
    weights = np.zeros(X.shape[1])
    previous_loss = compute_loss(X, y, weights)
    for i in range(max_iterations):
        weights = gradient_descent_step(
            X, y, weights, learning_rate, regularization, lam
        )
        loss = compute_loss(X, y, weights)
        # Stoppage critera - if previous_loss - loss is smaller than a set tollerance, then we stop our training
        # Tolerance set to 1e-4
        if abs(previous_loss - loss) < tolerance:
            break
        previous_loss = loss
    print(
        f"Logstic Regression: \nStopped after {i + 1} iterations Final loss = {loss:.4f}"
    )
    return weights


def evaluate(y_true, y_pred):
    # calculate accuracy, precision, and recall
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    false_positives = np.sum((y_pred == 1) & (y_true == 0))
    false_negatives = np.sum((y_pred == 0) & (y_true == 1))
    accuracy = np.mean(y_pred == y_true)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    return accuracy, precision, recall


def train_naive_bayes(X, y):
    vocab_size = X.shape[1]

    log_prior_spam = np.log(np.mean(y == 1))
    log_prior_ham = np.log(np.mean(y == 0))

    word_counts_spam = X[y == 1].sum(axis=0)
    word_counts_ham = X[y == 0].sum(axis=0)

    # Laplace smoothing, in log space: count(w,y) + 1 / Σcount(w',y) + |V| (here it is vocab + bias columnn)
    log_likelihood_spam = np.log(
        (word_counts_spam + 1) / (word_counts_spam.sum() + vocab_size)
    )
    log_likelihood_ham = np.log(
        (word_counts_ham + 1) / (word_counts_ham.sum() + vocab_size)
    )

    return log_prior_spam, log_prior_ham, log_likelihood_spam, log_likelihood_ham


def predict_naive_bayes(
    X, log_prior_spam, log_prior_ham, log_likelihood_spam, log_likelihood_ham
):
    # testing where y^ = argmax[logP(y)+ Σlog P(wi | y)]
    log_score_spam = log_prior_spam + X @ log_likelihood_spam
    log_score_ham = log_prior_ham + X @ log_likelihood_ham
    log_scores = np.column_stack([log_score_ham, log_score_spam])
    return np.argmax(log_scores, axis=1)


if __name__ == "__main__":
    X, y, vectorizer = load_data(ENRON_FILE_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train, X_test = add_bias_column(X_train), add_bias_column(X_test)

    # Logistic Regression
    # toggelable options for logistic regression
    choose_reg_option = ["l1", "l2"]

    for reg in choose_reg_option:
        weights = train_logistic_regression(
            X_train, y_train, learning_rate=0.01, regularization=reg, lam=0.01
        )
        y_pred = (predict_probability(X_test, weights) > 0.5).astype(int)
        accuracy, precision, recall = evaluate(y_test, y_pred)
        print(
            f"Reguralization = {reg} Accuracy = {accuracy:.4f} Precision = {precision:.4f} Recall = {recall:.4f} \n"
        )

    # Naive Bayes
    log_prior_spam, log_prior_ham, log_likelihood_spam, log_likelihood_ham = (
        train_naive_bayes(X_train, y_train)
    )
    y_pred = predict_naive_bayes(
        X_test, log_prior_spam, log_prior_ham, log_likelihood_spam, log_likelihood_ham
    )
    accuracy, precision, recall = evaluate(y_test, y_pred)
    print(
        f"Naive Bayes: \nAccuracy = {accuracy:.4f} Precision = {precision:.4f} Recall = {recall:.4f}"
    )
