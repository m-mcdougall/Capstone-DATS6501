# -*- coding: utf-8 -*-

#Non-Specific Imports
import os
import re
import pandas as pd
import numpy as np
import time
import concurrent.futures as cf
from tqdm import tqdm
import math


#Plotting Imports
import matplotlib.pyplot as plt
from  matplotlib.ticker import FuncFormatter
import seaborn as sns


#NLP Imports
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer


#Modeling Imports
import torch
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from datasets import load_metric
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler
from torch.utils.data import DataLoader
from torch.optim import AdamW



#Flatten list utility function
def flatten_list(list_in):
    return [item for sublist in list_in for item in sublist]


#Set the working directory
wd=os.path.abspath('C://Users//Mariko//Documents//GitHub//Capstone-DATS6501')
os.chdir(wd)

#%%
##############
#
#  Load in the features and predictions
#
##############


features = pd.read_csv(wd+'\\Data\\Features\\Sentences\\feature_sentence_matrix.csv', index_col=0)
predictions = pd.read_csv(wd+'\\Data\\Features\\Results\\ensemble_predictions.csv',index_col=0)


#Join features and predictions
features = features.join(predictions)
features = features.rename({'Prediction_Ensemble':"Prediction"}, axis =1)
features.head(10)

del predictions
#%%


# Now time to groupby for analysis
feature_val_overall=features.groupby(['Feature', 'Sentiment'])['Prediction'].mean()

feature_val_state = features.groupby(['Feature','State', 'Sentiment'])['Prediction'].mean()
feature_val_pand = features.groupby([ 'Feature','Pandemic_Timing','Sentiment'])['Prediction'].mean()


#%%



#Show the change from the mean for each state
delta_state = feature_val_state - feature_val_overall
#delta_pand = feature_val_pand - feature_val_overall



feature_val_pand_before = features[features.Pandemic_Timing == 'before']
feature_val_pand_after = features[features.Pandemic_Timing == 'after']
feature_val_pand_after = feature_val_pand_after.groupby([ 'Feature','Sentiment'])['Prediction'].mean()
feature_val_pand_before = feature_val_pand_before.groupby([ 'Feature','Sentiment'])['Prediction'].mean()

feature_val_pand = feature_val_pand_before - feature_val_pand_after


#%%

#Convert to change in strength of caring

pand_caring = feature_val_pand.reset_index()

care_more = pand_caring[((pand_caring.Sentiment =='POSITIVE')&(pand_caring.Prediction >0) ) |  ((pand_caring.Sentiment =='NEGATIVE')&(pand_caring.Prediction <-0) )]
care_less= pand_caring[((pand_caring.Sentiment =='NEGATIVE')&(pand_caring.Prediction >0) ) |  ((pand_caring.Sentiment =='POSITIVE')&(pand_caring.Prediction <-0) )]
care_same = pand_caring[pand_caring.Prediction == 0]


care_more['Prediction'] = care_more['Prediction'].apply(lambda x: abs(x))
care_less['Prediction'] = care_less['Prediction'].apply(lambda x: abs(x)*-1.0)


caring = pd.concat([care_more, care_less, care_same])



#%%
x2 = feature_val_pand.reset_index()
x2 = x2.sort_values('Prediction')
x2 = x2[((x2.Sentiment =='POSITIVE')&(x2.Prediction >0) ) |  ((x2.Sentiment =='NEGATIVE')&(x2.Prediction <-0) )]



fig = plt.figure(figsize=(14, 6))
sns.catplot(data=x2, y='Prediction', x='Feature', hue='Sentiment', kind='bar',
            height = 5, aspect=3,)
plt.show()



#%%

all_feats = features.Feature.unique()






