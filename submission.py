"""
This is an example script to generate the outcome variable given the input dataset.

This script should be modified to prepare your own submission that predicts 
the outcome for the benchmark challenge by changing the clean_df and predict_outcomes function.

The predict_outcomes function takes a Pandas data frame. The return value must
be a data frame with two columns: nomem_encr and outcome. The nomem_encr column
should contain the nomem_encr column from the input data frame. The outcome
column should contain the predicted outcome for each nomem_encr. The outcome
should be 0 (no child) or 1 (having a child).

clean_df should be used to clean (preprocess) the data.

run.py can be used to test your submission.
""" 

# List your libraries and modules here. Don't forget to update environment.yml!
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from joblib import dump
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from joblib import load
import seaborn as sns


def clean_df(df, background_df=None):
    if isinstance(df, str):
        df = pd.read_csv(df, sep=',', low_memory=False)
        
    df = df[['nomem_encr', #Unique number
                    'outcome_available', #1 = outcome is available --> merge with outcome dataset
                    'cf17j024', 'cf18k024', 'cf20m024', 'cf20m025', 'cf08a026', 'cf14g026',
                    'cf17j026', 'cf20m026', 'cf20m030', 'cf20m032', 'cf17j128', 'cf18k128',
                    'cf19l128', 'cf20m128', 'cf17j129', 'cf18k129', 'cf19l129', 'cf20m129',
                    'cf14g130', 'cf16i130', 'cf17j130', 'cf18k130', 'cf19l130', 'cf20m130',
                    'cf17j180', 'cf18k180', 'cf19l180', 'cf20m180', 'cf20m181', 'cf18k432',
                    'cf19l432', 'cf18k454', 'cf20m454', 'cf20m455', 'cf20m471', 'cf17j483',
                    'cf18k483', 'cf20m483', 'cf17j484', 'cf18k484', 'cf20m484', 'cf17j485',
                    'cf19l485', 'cf20m485', 'cf17j486', 'cf18k486', 'cf19l486', 'cf20m486',
                    'cf18k487', 'cf19l487', 'cf20m487', 'cf18k488', 'cf20m488', 'cf20m526',
                    'cf20m527', 'cf20m528', 'cf20m529', 'cf20m530', 'ci17j006', 'ci18k006',
                    'ci19l006', 'ci20m006', 'gender_imp', 'birthyear_imp', 'oplzon',
                    'oplmet', 'woonvorm', 'woning', 'brutoink_f'
             ]].copy()
             

    df = df[df['outcome_available'] == 1]
    
    df.fillna(0, inplace=True)
    
    if background_df is not None:
        if isinstance(background_df, str):
            background_df = pd.read_csv(background_df, sep=',', low_memory=False)
        
        if 'birthyear_imp' in background_df.columns:
            background_df = background_df[['nomem_encr', 'gender_imp', 'birthyear_imp', 'oplzon', 'oplmet', 'woonvorm', 'woning', 'brutoink_f']].copy()
            background_df = background_df[(background_df['birthyear_imp'] >= 1975) & (background_df['birthyear_imp'] <= 2002)]
            background_df = background_df.groupby('nomem_encr').agg({
                'gender_imp'    : 'last',
                'birthyear_imp' : 'last',
                'oplzon'        : 'last',
                'oplmet'        : 'last', 
                'woonvorm'      : 'last', 
                'woning'        : 'last',
                'brutoink_f'    : 'mean'}).reset_index()
        
            background_df = background_df.drop_duplicates()
        
    df = pd.merge(df, background_df, on="nomem_encr", how="left")
    df.fillna(0, inplace=True) #removed all the NaN values
    
    return df

# df = pd.read_csv('PreFer_fake_data.csv', sep=',', low_memory=False)
# background_df = pd.read_csv('PreFer_fake_background_data.csv', sep=',', low_memory=False)
# merged_train_background_data_df = clean_df(df, background_df)
# print(merged_train_background_data_df)

# print(merged_train_background_data_df.isnull().sum()) #checked for NaN values
# # print(merged_train_background_data_df.shape)
# # print(merged_train_background_data_df.head())

# print(merged_train_background_data_df.dtypes)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from joblib import load

def predict_outcomes(df, background_df=None, model_path="model.joblib"):
    
    if "nomem_encr" not in df.columns:
        print("The identifier variable 'nomem_encr' should be in the dataset")
    
    model = joblib.load(model_path)
    # print(model)
    
    # Preprocess the fake / holdout data
    df = clean_df(df, background_df)
    # print(df)
    
    # Exclude the variable nomem_encr if this variable is NOT in your model
    vars_without_id = df.columns[df.columns != 'nomem_encr']
    # print(vars_without_id) --> Make this print function work, return something else --> return 1 (and not return df_predict)

    # Generate predictions from model, should be 0 (no child) or 1 (had child)
    predictions = model.predict(df[vars_without_id])
    # print(predictions)

    # Output file should be DataFrame with two columns, nomem_encr and predictions
    df_predict = pd.DataFrame(
            {"nomem_encr": df["nomem_encr"], "prediction": predictions})

    # Return only dataset with predictions and identifier
    return df_predict


# # load the data
# fake = pd.read_csv("PreFer_fake_data.csv") 
# fake_background = pd.read_csv('PreFer_fake_background_data.csv', sep=',', low_memory=False)
# predict_outcomes(fake, fake_background)





