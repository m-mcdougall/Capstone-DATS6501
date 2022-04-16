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

#Get all features

pre_pandemic_features = pd.read_csv(wd+'\\Data\\Tfidf\\PrePandemic_reviews.csv', index_col=0).iloc[0:150, :]
post_pandemic_features = pd.read_csv(wd+'\\Data\\Tfidf\\PostPandemic_reviews.csv', index_col=0).iloc[0:150, :]
#%%

#Prepandemic
pre_pandemic_features = pre_pandemic_features.reset_index()
pre_pandemic_features = pre_pandemic_features.rename({'index':'Rank_Pre'}, axis=1)
pre_pandemic_features = pre_pandemic_features.drop('TFIDF', axis=1)


#Postpandemic
post_pandemic_features = post_pandemic_features.reset_index()
post_pandemic_features = post_pandemic_features.rename({'index':'Rank_Post'}, axis=1)
post_pandemic_features = post_pandemic_features.drop('TFIDF', axis=1)






















