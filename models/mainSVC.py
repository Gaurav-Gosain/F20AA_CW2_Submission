import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
import preprocessor as pp
import re
import spacy
from nltk.stem.snowball import SnowballStemmer
# from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
# Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
# Gausian Naive Bayes
# SVM
from sklearn.svm import SVC
# Linear Model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score
import seaborn as sns
import threading
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import multiprocessing


dfLematized = pd.read_csv('data/preprocessed_data.csv')
# typecasting cleaned_text to string
dfLematized['l_text'] = dfLematized['l_text'].astype(str)
dfLematized['s_text'] = dfLematized['s_text'].astype(str)

print("Helper functions to train and test the models")


def prockfold(dfLematized, index,model, numberOfFolds, optionsName, nGram):
    # 5 fold cross validation
    kfold = StratifiedKFold(n_splits=numberOfFolds,
                            shuffle=True, random_state=7)
    totalAccuracy = 0
    totalFScore = 0
    totalConfusion_matrix = None
    threads = []
    for train_index, test_index in kfold.split(dfLematized[index], dfLematized['Score']):
        X_train, X_test = dfLematized.iloc[train_index][index], dfLematized.iloc[test_index][index]
        y_train, y_test = dfLematized.iloc[train_index]['Score'], dfLematized.iloc[test_index]['Score']
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        totalAccuracy += accuracy_score(y_test, y_pred)
        totalFScore += f1_score(y_test, y_pred, average='macro')
        totalConfusion_matrix = totalConfusion_matrix + confusion_matrix(
            y_test, y_pred) if totalConfusion_matrix is not None else confusion_matrix(y_test, y_pred)

    fscore = totalFScore/kfold.get_n_splits()
    acc_score = totalAccuracy/kfold.get_n_splits()
    confusion_matrix_result = totalConfusion_matrix/kfold.get_n_splits()
    moddddeeell = {"accuracy": acc_score, "f1_score": fscore}

    # Save the confusion matrix np save
    with open(f"unbalanced_{model['model'].__class__.__name__}_{model['vectorizer'].__class__.__name__}_{optionsName}_{nGram}_confusion_matrix.npy", "wb") as of:
        np.save(of, confusion_matrix_result)
    # Save the file as json
    with open(f"unBalanced_{model['model'].__class__.__name__}_{model['vectorizer'].__class__.__name__}_{optionsName}_{nGram}.json", 'w') as f:
        json.dump(moddddeeell, f)

def main():

    numberOfFolds = 5
    processes = []
    # To compare vectorizers
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(1, 1))),
        ('model', SVC(kernel="sigmoid",gamma="scale", tol=0.1))
    ])

    p = multiprocessing.Process(target=prockfold, args=(dfLematized, 'l_text', pipeline, numberOfFolds, 'Lematization', 'Unigram'))
    processes.append(p)
    p.start()

    # To compare vectorizers
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(ngram_range=(1, 1))),
        ('model', SVC(kernel="sigmoid",gamma="scale", tol=0.1))
    ])
    p = multiprocessing.Process(target=prockfold, args=(dfLematized, 'l_text', pipeline, numberOfFolds, 'Lematization', 'Unigram'))
    processes.append(p)
    p.start()

    # To compare Normalization techniques
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(1, 1))),
        ('model', SVC(kernel="sigmoid",gamma="scale", tol=0.1))
    ])
    p = multiprocessing.Process(target=prockfold, args=(dfLematized, 's_text', pipeline, numberOfFolds, 'Stemming', 'Unigram'))
    processes.append(p)
    p.start()

    # To compare n-grams
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(2, 2))),
        ('model', SVC(kernel="sigmoid",gamma="scale", tol=0.1))
    ])
    p = multiprocessing.Process(target=prockfold, args=(dfLematized, 'l_text', pipeline, numberOfFolds, 'Lematization', 'Bigram'))
    processes.append(p)
    p.start()

    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(3, 3))),
        ('model', SVC(kernel="sigmoid",gamma="scale", tol=0.1))
    ])
    p = multiprocessing.Process(target=prockfold, args=(dfLematized, 'l_text', pipeline, numberOfFolds, 'Lematization', 'Trigram'))
    processes.append(p)
    p.start()

    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(1, 2))),
        ('model', SVC(kernel="sigmoid",gamma="scale", tol=0.1))
    ])
    p = multiprocessing.Process(target=prockfold, args=(dfLematized, 'l_text', pipeline, numberOfFolds, 'Lematization', 'Unigram and Bigram'))
    processes.append(p)
    p.start()

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
