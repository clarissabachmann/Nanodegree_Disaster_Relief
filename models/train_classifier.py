"""import libraries"""
import sys
import re
import pandas as pd
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
    Load in data from cleaned database and create Y and X inputs
    Also return category names to use with model evaluation
    """
    engine = create_engine("sqlite:///{}".format(database_filepath))
    df = pd.read_sql_query("SELECT * from message", engine)
    X=df['message']
    y= df[['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue', 'security', 'military',
      'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related',
      'transport', 'buildings', 'electricity', 'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure', 'weather_related',
      'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report']]
    category_names = df[['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue', 'security', 'military',
      'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related',
      'transport', 'buildings', 'electricity', 'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure', 'weather_related',
      'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report']]
    return X, y, category_names


def tokenize(text):
    """
    Tokenize the input catastrophe messages to separate sentences into words
    Lematize to reduce words to their roots (to closest noun)
    Also clean tokens by making them lower case and removing leading and trailing spaces
    Return cleaned tokens
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
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model
    """
    y_pred = model.predict(X_test)
    return y_pred


def save_model(model, model_filepath):
    """
    save model as a pickle file
    """
    pickle.dump(model, open(model_filepath, "wb"))

    
def main():
    """
    Function that runs the whole model building function
    Produces an output to show how far along model build is
    Also contains instructions for model build
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, Y_test, X_test, category_names)

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
