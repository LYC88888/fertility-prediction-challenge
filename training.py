import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from joblib import dump
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

# loading data (predictors)
train = pd.read_csv("PreFer_train_data.csv", low_memory = False) 
# loading the outcome
outcome = pd.read_csv("PreFer_train_outcome.csv") 

def train_save_model(cleaned_df, outcome_df):
    
    if isinstance(outcome_df, str):
        outcome_df = pd.read_csv(outcome_df, sep=',', low_memory=False)
        
    # Merge the cleaned dataframe with outcome dataframe on 'nomem_encr'
    model_df = pd.merge(cleaned_df, outcome_df, on="nomem_encr")
    
    # Drop any rows where 'new_child' might be NaN to ensure clean training data
    model_df = model_df.dropna(subset=['new_child'])
    
    # Prepare the feature matrix X by dropping the target variable and other non-predictor columns
    X = model_df.drop(['new_child', 'nomem_encr'], axis=1) #axis = 1 equals to columns / axis = 0 equals to rows
    # The target variable y is what we want to predict
    y = model_df['new_child']

    # Split the data into training (90%) and validation sets (10%)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    # Apply SMOTE to address class imbalance by oversampling the minority class in the training data
    smote = SMOTE(random_state=42)
    X_train_SMOTE, y_train_SMOTE = smote.fit_resample(X_train, y_train)

    # Initialize the XGBoost classifier with specific hyperparameters
    model = xgb.XGBClassifier(objective='binary:logistic', learning_rate=0.1, n_estimators=90, use_label_encoder=False, eval_metric='logloss')

    # Train the model on the oversampled training data
    model.fit(X_train_SMOTE, y_train_SMOTE)
    
    # Save the trained model to a file for later use
    joblib.dump(model, "model.joblib")



import os
# print(os.getcwd())
# os.chdir(path to your local repository) #<---- provide the path here

# preprocessing the data
train_cleaned = clean_df(train, background_df)

# training and saving the model
train_save_model(train_cleaned, outcome)


