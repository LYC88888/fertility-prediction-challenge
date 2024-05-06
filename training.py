import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from joblib import dump
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_recall_curve, f1_score


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
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    
    # With specific hyperparameters
    model = xgb.XGBClassifier(objective='binary:logistic', learning_rate=0.1, n_estimators=50, use_label_encoder=False, eval_metric='logloss')

    # Train the model on the oversampled training data
    model.fit(X_train, y_train)
    
    # Predict probabilities
    y_probs_XGB_threshold = model.predict_proba(X_val)[:, 1]

    # Calculate precision, recall, and thresholds
    precision, recall, thresholds = precision_recall_curve(y_val, y_probs_XGB_threshold)

    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall)

    # Locate the index of the largest F1 score
    ix = np.argmax(f1_scores)
    # print('Best Threshold=%f, F1-Score=%.3f' % (thresholds[ix], f1_scores[ix]))
    
    # Using threshold to convert probabilities to binary class predictions
    y_pred_XGB_threshold = (y_probs_XGB_threshold >= thresholds[ix]).astype(int)
    
    # print('Accuracy:', accuracy_score(y_val, y_pred_XGB_threshold))
    # print('XGBoost without SMOTE + Threshold:')
    # print(classification_report(y_val, y_pred_XGB_threshold))
    
    # Save the trained model to a file for later use
    joblib.dump(model, "model.joblib")

# REAL DATA:
import os
# print(os.getcwd())
# os.chdir(path to your local repository) #<---- provide the path here

# preprocessing the function
train_cleaned = clean_df(df, background_df)

# training and saving the model
train_save_model(train_cleaned, outcome)
