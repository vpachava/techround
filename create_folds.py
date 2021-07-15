# create_folds.py
import pandas as pd
from sklearn import model_selection


dataset = pd.read_csv("../inputs/Acc.csv", low_memory=False)
## making binary targets due to 1 is police attended scene of accident and 2 is not and 3 is manual reported incident
dataset['Did_Police_Officer_Attend_Scene_of_Accident']= dataset['Did_Police_Officer_Attend_Scene_of_Accident'].apply(lambda x: 1 if x==1 else 0)
 # we create a new column called kfold and fill it with -1
dataset["kfold"] = -1

 # the next step is to randomize the rows of the data
dataset = dataset.sample(frac=1).reset_index(drop=True)

 # fetch labels
y = dataset.Did_Police_Officer_Attend_Scene_of_Accident.values

 # initiate the kfold class from model_selection module
kf = model_selection.StratifiedKFold(n_splits=5)

 # fill the new kfold column
for f, (t_, v_) in enumerate(kf.split(X=dataset, y=y)):
    dataset.loc[v_, 'kfold'] = f

if __name__ == "__main__" :
    # save the new csv with kfold column
    dataset.to_csv("../inputs/train_folds.csv", index=False)