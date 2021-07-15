# try to run xgboost with numerical columns inclusion.

# lbl_xgb.py
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn import preprocessing

def run(fold):
    
    # load the full training data with folds
    dataset = pd.read_csv("../inputs/train_folds.csv")
    # list of numerical columns
    num_cols = [
        "Number_of_Vehicles",
        "Number_of_Casualties",
        "Speed_limit",        
        "1st_Road_Number",
        "2nd_Road_Number"
        ]
    # drop numerical columns
    #dataset = dataset.drop(num_cols, axis=1)
    # drop id and other relavant not usefull columns
    # Accident_Index, Longitude, Latitude, Date, Time , 
    cols_to_drop = ["Accident_Index", "Longitude", "Latitude", "Date", "Time", "Local_Authority_(Highway)","Local_Authority_(District)","LSOA_of_Accident_Location","Location_Easting_OSGR","Location_Northing_OSGR"]
    dataset = dataset.drop(cols_to_drop, axis=1)
    # map targets to 0s and 1s
    target_mapping = {
     ">1": 0,
     "=1": 1
    }
    dataset.loc[:, "Did_Police_Officer_Attend_Scene_of_Accident"] = dataset.Did_Police_Officer_Attend_Scene_of_Accident.map(target_mapping)
    # all columns are features except kfold & income columns
    features = [
        f for f in dataset.columns if f not in ("kfold", "Did_Police_Officer_Attend_Scene_of_Accident")
        ]
    # fill all NaN values with NONE
    # note that I am converting all columns to "strings"
    # it doesnt matter because all are categories
    for col in features:
        dataset.loc[:, col] = dataset[col].astype(str).fillna("NONE")
        
    # now its time to label encode the features
    for col in features:
        if col not in num_cols:
            # initialize LabelEncoder for each feature column
            lbl = preprocessing.LabelEncoder()
            
            # fit label encoder on all data
            lbl.fit(dataset[col])
            # transform all the data
            dataset.loc[:, col] = lbl.transform(dataset[col])
    
       
    # get training data using folds
    df_train = dataset[dataset.kfold != fold].reset_index(drop=True)
    # get validation data using folds
    df_valid = dataset[dataset.kfold == fold].reset_index(drop=True)
    # get training data
    x_train = df_train[features].values
    # get validation data
    x_valid = df_valid[features].values
    # initialize xgboost model
    model = xgb.XGBClassifier(
        n_jobs=-1,
        max_depth=7,
        use_label_encoder=False
        )
    # fit model on training data (ohe)
    model.fit(x_train, df_train.Did_Police_Officer_Attend_Scene_of_Accident.values)
    # predict on validation data
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds = model.predict_proba(x_valid)[:, 1]
    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.Did_Police_Officer_Attend_Scene_of_Accident.values, valid_preds)
    # print auc
    print(f"Fold = {fold}, AUC = {auc}") 


if __name__ == "__main__" :

    for fold_ in range(5):
        run(fold_)   