import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Function used to read in raw data in the form of csv file
    Then merge them together on Id
    return pandas dataframe
    """
    dfMessages = pd.read_csv(messages_filepath)
    dfCategories = pd.read_csv(categories_filepath)
    df = dfMessages.merge(dfCategories, on=["id"])
    return dfCatastropheMessages


def clean_data(dfCatastropheMessages):
    """
    Use dataframe read in by Load_data function
    Split up incomming messages by ;
    Select the column headers
    Then use the last two values of the me 
    """
    categories = df["categories"].str.split(";", expand=True)
    row = categories.iloc[0]
    category_colnames = row.map(lambda x: str(x)[:-2])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].map(lambda x: str(x)[-1])
        categories[column] = pd.to_numeric(categories[column])
    df = df.drop(columns=["categories"])
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    df = df.loc[df['related'] != 2]
    return df
  
def save_data(df, database_filename):
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('message', engine, index=False) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        dfFinal = clean_data(dfCatastrophyMessages)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(dfFinal, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
