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
import joblib 
from joblib import load
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from joblib import dump
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def clean_df(df, background_df=None):
    if isinstance(df, str):
        df = pd.read_csv(df, sep=',', low_memory=False)
        
    df = df[['nomem_encr', #Unique number
             'outcome_available', #1 = outcome is available --> merge with outcome dataset
             'cf08a024', 'cf09b024', 'cf10c024', 'cf11d024', 'cf12e024', 'cf13f024', 'cf14g024', 'cf15h024', 'cf16i024', 'cf17j024', 'cf18k024', 'cf19l024', 'cf20m024', #5 Do you currently have a partner? 
             'cf08a025', 'cf09b025', 'cf10c025', 'cf11d025', 'cf12e025', 'cf13f025', 'cf14g025', 'cf15h025', 'cf16i025', 'cf17j025', 'cf18k025', 'cf19l025', 'cf20m025', #6 Do you live together with this partner? 
             'cf08a026', 'cf09b026', 'cf10c026', 'cf11d026', 'cf12e026', 'cf13f026', 'cf14g026', 'cf15h026', 'cf16i026', 'cf17j026', 'cf18k026', 'cf19l026', 'cf20m026', #7 What is his or her year of birth? 
             'cf08a030', 'cf09b030', 'cf10c030', 'cf11d030', 'cf12e030', 'cf13f030', 'cf14g030', 'cf15h030', 'cf16i030', 'cf17j030', 'cf18k030', 'cf19l030', 'cf20m030', #10 Are you married to this partner? 
             'cf08a032', 'cf09b032', 'cf10c032', 'cf11d032', 'cf12e032', 'cf13f032', 'cf14g032', 'cf15h032', 'cf16i032', 'cf17j032', 'cf18k032', 'cf19l032', 'cf20m032', #11 What is your partner's gender? 
             'cf08a035', 'cf09b035', 'cf10c035', 'cf11d035', 'cf12e035', 'cf13f035', 'cf14g035', #12 Have you had any children? 
             'cf08a128', 'cf09b128', 'cf10c128', 'cf11d128', 'cf12e128', 'cf13f128', 'cf14g128', 'cf15h128', 'cf16i128', 'cf17j128', 'cf18k128', 'cf19l128', 'cf20m128', #15 Do you think you will have children in the future? 
             'cf08a129', 'cf09b129', 'cf10c129', 'cf11d129', 'cf12e129', 'cf13f129', 'cf14g129', 'cf15h129', 'cf16i129', 'cf17j129', 'cf18k129', 'cf19l129', 'cf20m129', #16 How many children do you think you will have in the future? 
             'cf08a130', 'cf09b130', 'cf10c130', 'cf11d130', 'cf12e130', 'cf13f130', 'cf14g130', 'cf15h130', 'cf16i130', 'cf17j130', 'cf18k130', 'cf19l130', 'cf20m130', #17 Within how many years do you hope to have your (first-next) child? 
             'cf08a180', 'cf09b180', 'cf10c180', 'cf11d180', 'cf12e180', 'cf13f180', 'cf14g180', 'cf15h180', 'cf16i180', 'cf17j180', 'cf18k180', 'cf19l180', 'cf20m180', #29 How satisfied are you with your current relationship? 
             'cf08a181', 'cf09b181', 'cf10c181', 'cf11d181', 'cf12e181', 'cf13f181', 'cf14g181', 'cf15h181', 'cf16i181', 'cf17j181', 'cf18k181', 'cf19l181', 'cf20m181', #30 How satisfied are you with your family life? 
             'cf08a187', 'cf09b187', 'cf10c187', 'cf11d187', 'cf12e187', 'cf13f187', 'cf14g187', #35 How is the household work divided between you and your partner? - preparing food 
             'cf08a188', 'cf09b188', 'cf10c188', 'cf11d188', 'cf12e188', 'cf13f188', 'cf14g188', #36 How is the household work divided between you and your partner? - laundry, ironing 
             'cf08a189', 'cf09b189', 'cf10c189', 'cf11d189', 'cf12e189', 'cf13f189', 'cf14g189', #37 How is the household work divided between you and your partner? - house cleaning 
             'cf08a190', 'cf09b190', 'cf10c190', 'cf11d190', 'cf12e190', 'cf13f190', 'cf14g190', #38 How is the household work divided between you and your partner? - odd jobs in and around the house 
             'cf08a191', 'cf09b191', 'cf10c191', 'cf11d191', 'cf12e191', 'cf13f191', 'cf14g191', #39 How is the household work divided between you and your partner? - financial administration 
             'cf08a192', 'cf09b192', 'cf10c192', 'cf11d192', 'cf12e192', 'cf13f192', 'cf14g192', #40 How is the household work divided between you and your partner? - grocery shopping 
             'cf09b402', 'cf10c402', 'cf11d402', 'cf12e402', 'cf13f402', 'cf14g402', 'cf15h402', 'cf16i402', 'cf17j402', 'cf18k402', 'cf19l402', 'cf20m402', #46 Is this partner the same partner you entered in the questionnaire last year? 
             'cf09b406', 'cf10c406', 'cf11d406', 'cf12e406', 'cf13f406', 'cf14g406', #49 Are all these children still alive? 
             'cf09b407', 'cf10c407', 'cf11d407', 'cf12e407', 'cf13f407', 'cf14g407', 'cf15h407', 'cf16i407', 'cf17j407', 'cf18k407', 'cf19l407', 'cf20m407', #50 Do you consider yourself as childless by choice, or would you have liked to have children? 
             'cf09b408', 'cf10c408', 'cf11d408', 'cf12e408', 'cf13f408', 'cf14g408', 'cf15h408', 'cf16i408', 'cf17j408', 'cf18k408', 'cf19l408', 'cf20m408']].copy() #51 Do you consider it a loss not having had children, or does it not matter much, or are you content with it? 

    df = df[df['outcome_available'] == 1]
    
    df.fillna(0, inplace=True)
    
    if background_df is not None:
        if isinstance(background_df, str):
            background_df = pd.read_csv(background_df, sep=',', low_memory=False)
        
        if 'birthyear_imp' in background_df.columns:
            background_df = background_df[['nomem_encr', 'gender_imp', 'birthyear_imp', 'brutoink_f']].copy()
            background_df = background_df[(background_df['birthyear_imp'] >= 1975) & (background_df['birthyear_imp'] <= 2002)]
            background_df = background_df.groupby('nomem_encr').agg({
                'gender_imp': 'first',
                'birthyear_imp': 'first',
                'brutoink_f': 'mean'
            }).reset_index()
        
            background_df = background_df.drop_duplicates()
        
    df = pd.merge(df, background_df, on="nomem_encr", how="left")
    df.fillna(0, inplace=True) #removed all the NaN values
    
    return df

# df = pd.read_csv('PreFer_fake_data.csv', sep=',', low_memory=False)
# background_df = pd.read_csv('PreFer_fake_background_data.csv', sep=',', low_memory=False)
# merged_train_background_data_df = clean_df(df, background_df)
# print(merged_train_background_data_df)


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
# predict_outcomes(fake, background_df)

