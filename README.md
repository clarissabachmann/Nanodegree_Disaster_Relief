Nanodegree Disaster Relief

Project Summary

This repository contains the code used to create a machine learning NLP (natural language processing) pipeline in order to categorize disaster messages into appropriate categories to allow more rapid understanding of a messages content and therefore its relevance. This is done in three steps. The initial step reads in the data (also stored in the repository) and runs an ETL pipeline to clean and prepare the data for modelling and then write it to a SQLite database, as found in the data folder. The second step is found in the modelling folder where data is tokenize and lemmatized, a pipeline built and parameters optimized with GridsearchCV. The model is then fit, evaluated, and saved as a pickle file in the model folder. Finally the app folder contains the required html templates for the flask webapp and the run.py folder that prepares the model, data, and graphs in order to deploy the webapp. The webapp shows a graph of the number of entries in each category from the test data used to train the model and allows a user to input a message and see what category the model considers it to belong to.

Packages Used

ETL packages

import sys: access variables
import pandas as pd: manipulate data in a dataframe format
from sqlalchemy import create_engine: save data in SQLlite

Model Packages

import sys: access variables
import re:for regex changes
import pandas as pd: manipulate data in a dataframe format
from nltk.tokenize import word_tokenize: to break up sentences into word tokens
from nltk.stem import WordNetLemmatizer: to reduce words to their lemmas

from sklearn.model_selection import GridSearchCV: to optimize parameters
from sklearn.ensemble import RandomForestClassifier: to run a random forest model
from sklearn.model_selection import train_test_split: split data into test and train groups
from sklearn.pipeline import Pipeline: to create the model pipeline
from sklearn.multioutput import MultiOutputClassifier: used in order to run random forest classifier over multiple outputs
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer: to convert text into matrix of tokens and then transform count of matrix to a normalized term frequency times times inverse term frequency.
from sqlalchemy import create_engine: read in data from database
import pickle: to save te model

import nltk: python library for natural language processing. Needed to prepare and use data.
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger']): download necessary components

App packages

import json: to be able to use json dataformat
import plotly: to plot data in webapp
import pandas as pd: in order to work with data in dataframes
from nltk.stem import WordNetLemmatizer: to lemmatize input
from nltk.tokenize import word_tokenize: to reduce input to tokens
from flask import Flask: to build webapp
from flask import render_template, request, jsonify: build webapp
from plotly.graph_objs import Bar: to build bar plots
from sklearn.externals import joblib: for model usage
from sqlalchemy import create_engine: to import data

Data Folder

The data folder contains two csvs, the disaster_categories.csv and the disaster_messages.csv. These are the raw datafiles that are used in the building of this model. It also contains a database file (DisasterResponse.db) that contains the cleaned output of the process_data.py code also found in this folder. This is used in training and testing the model. The process_data.py alters the code in the following way:
  Read in the csvs
  Merge two csvs together
  Use each category listed to create new columns and use 0 or 1 as the row value to show whether a message belongs to a category or not
  Concatenate the new columns with the original dataframe and drop the original categories column
  Drop all duplicates
  Remove rows where values are not 0 or 1
  Write data into SQLite database

Model Folder

The modle folder contains the train_classifier.py code file and the classifier.pkl that is created by train_classifier.py code. The train_classifier.py follows these steps to build the model:
  First the data from the database is read in into X (the messages) values, and Y values (the categories) and the category names are       defined
  Then the train/test split is defined
  Then the build pipeline is defined where initially the messages are turned into tokens, lemmatized, and cleaned to be turned into a     matrix of tokens. Then the matrix is transformed to a normalized term frequency times times inverse term frequency. Then a               MultiOutputClassifier random forest model is fit. Then parameters are defined. Finally a GridsearchCV is run using the model pipeline   and defined parameters to optimize parameters

App folder

The app folder contains the run.py file but also a second folder containing the go.html and master.html files used as templates for the flask webapp. The run.py folder creates the webapp by:
  Reading in the model from the pickle file and the database data
  Then creates a bar plot for the homepage to show how many messages belong to each category in the training and test data
  Then the plots are encored in json and rendered into the webpage
  Then it creates the component of the webpage that allows for user input of a message that is then run through the model
  Finally it deploys the webapp

Getting Started

To run the code held in this repository follow the following steps:
First you have to run the process_data.py file found in the data folder. Open the command prompt or terminal, navigate to the appropriate folder where the code file and data are stored and the run: python process_data.py disaster_messages.csv disaster_categories.csv. It is only necessary to run this once in order to create the database. If this already exists then it is not required to run this.
Then you have to run the train_classifier.py file found in the models folder. To run this piece of code, open the command prompt or terminal, navigate to the appropriate folder where the code file and data are stored and the run: python train_classifier.py ../data/{DatabaseName}.db {pickleName}.pkl. This must only be run once and if the pickle file already exists then it does not have to be run
Finally you have to run the run.py file found in the app folder. To deploy the webapp first open a terminal/command prompt and navigate to relevant folder. Then run env|grep WORK in order to get your ID and url. Then open another terminal and run the run.py code. After this is successfully run open a browser window and type {spaceID}-3001.udacity-student-workspace.com into the url bar to access the webpage

Files in the repository

The following files are found in the repository:
data folder:

process_data.py file that runs the ETL and saves it as a SQLite database
disaster_messages.csv: one of the raw data files
disaster_categories.csv: the other raw data file
DisasterResponse.db: the outputted database created by the process_data.py code
models folder:

train_classifier.py: file that trains and fits the model and saves it as a pickle file
classifier.pkl: pickle file created by train_classifier
app folder:

run.py: file used to create and deploy the webapp
templates foler: contains the go.html and master.html files used by run.py to create the webapp
All components mentioned here are used together in order to produce the webapp and model used.
(templates from udacity were used)
