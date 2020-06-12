"""import libraries"""
import sys
import re
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine
from sklearn.metrics import classification_report
import pickle
from sklearn.utils.multiclass import type_of_target

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

def load_data(database_filepath):
    """
    Load in data from cleaned database and create y and X inputs
    Also return category
    names to use with model evaluation
    input: database_filepath
    output: X, y, and category names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('message',engine)
    X = df.iloc[:,1]
    y = df.iloc[:,4:]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """
    Tokenize the input catastrophe messages to separate sentences into words
    Lematize to reduce words to their roots (to closest noun)
    Also clean tokens by making them lower case and removing leading and trailing spaces

    input: text from messages
    output: cleaned tokens 
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Create model build pipline
    Then specify parameters
    Run gridsearch to select optimal parameters
    input: nothing
    output: model
    """
    model = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])


    parameters = {
        'clf__estimator__n_estimators': [1, 200],
        'clf__estimator__min_samples_split': [2, 100], 
        'clf__estimator__min_samples_leaf': [5, 100]
    }
    cv = GridSearchCV(estimator=model, param_grid=parameters, cv=3)
    return model

def evaluate_model(model, X_train, Y_train, y_test, X_test, category_names):
    """
    Evaluate model and return the classification report
    Both fit and predict were combined into one function to maintain data structure (does not work separated)
    input: model, X_train, Y_train, y_test, X_test, category_names
    output: classfication report
    """
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=category_names)) 


def save_model(model, model_filepath):
    """
    save model as a pickle file
    input: model and model__filepath to save model to
    output: nothing
    """
    pickle.dump(model, open(model_filepath, "wb"))

    
def main():
    """
    Function that runs the whole model building function
    Produces an output to show how far along model build is
    Also contains instructions for model build
    input: database_filepath and model_filepath
    output: nothing
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()

        print('Train and Evaluating model...')
        evaluate_model(model, X_train, Y_train, y_test, X_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
