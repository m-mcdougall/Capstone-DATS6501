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


#Positive and negative words taken from merriam-Webster.
#Positive words are from the synonyms for "Good"
#Negative words are from the synonyms for "Bad"
#Definitions were selective - Eg, morally good was not included
#Some synonyms removed if very non-applicable or slang (eg "jake" for good)



words_pos = list(set(pd.read_csv(wd+'//PositiveWords.txt', sep = ',', header=None).iloc[0,:].array))
words_neg = list(set(pd.read_csv(wd+'//NegativeWords.txt', sep = ',', header=None).iloc[0,:].array))


#%%

#Make sure all copied words have the correct sentiment

from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model = 'distilbert-base-uncased-finetuned-sst-2-english')


#Positive
collector = []
for word in words_pos:
    collector.append([word, sentiment_pipeline(word)[0]['label'], sentiment_pipeline(word)[0]['score']])

collector_pos = pd.DataFrame(collector, columns=['Word', 'Sentiment', 'Score'])


#Negative
collector = []
for word in words_neg:
    collector.append([word, sentiment_pipeline(word)[0]['label'], sentiment_pipeline(word)[0]['score']])

collector_neg= pd.DataFrame(collector, columns=['Word', 'Sentiment', 'Score'])


#%%

#Remove incorrectly sentimented words

print('    Words Removed \n  Positive Sentiments:\n---------------------------\n')
print(collector_pos[collector_pos['Sentiment'] =='NEGATIVE'])
print('\n-----------------------------------------\n')

collector_pos = collector_pos[collector_pos['Sentiment'] !='NEGATIVE']


print('    Words Removed \n  Negative Sentiments:\n---------------------------\n')
print(collector_neg[collector_neg['Sentiment'] =='POSITIVE'])
print('\n-----------------------------------------\n')

collector_neg = collector_neg[collector_neg['Sentiment'] !='POSITIVE']

#%%

# Sample the words to select our sentiment sample

sample_pos = collector_pos.sample(n=25, replace=False, random_state=42)
sample_neg =collector_neg.sample(n=25, replace=False, random_state=42)

sample_sentiment = pd.concat([sample_pos,sample_neg])
sample_sentiment.reset_index(drop=True, inplace=True)
sample_sentiment = sample_sentiment.drop('Score', axis=1)


#%%

#Get all state abbreviations used in the project

#Loop through all cities - Collect all Hotels
all_files = os.listdir(wd+'\\Data\\Cleaned\\')
all_cities = [file for file in all_files if '.pkl' in file and 'Hotels' in file]

#Start with just one
#all_cities = [all_cities[1]]

hotels = []
for city in all_cities:
    hotels_file =  pd.read_pickle(wd+'\\Data\\Cleaned\\'+city)
    
    #Drop some columns to make the dataframe a bit lighter
    hotels_file.drop(['hotel_city','hotel_blurb', 'hotel_prop_amenity',
                      'hotel_room_amenity','hotel_location_food', 'hotel_location_attract'], 
                     axis=1, inplace=True)
    
    hotels.append(hotels_file)

hotels_df = pd.concat(hotels)

#Delete intermediaries to free up memory
del all_files, all_cities, city, hotels, hotels_file



## Engineer the state from the address

#Split the address by commas
new = hotels_df.hotel_address.str.split(',')

#State and Zip Code in the final comma set (total number varies)
new = new.apply(lambda x: x[-1])
#Remove zip code
new = new.str.replace(r'[ ][\d-]+', '', regex=True)
new = new.str.replace(r'[ ]', '', regex=True)

states = list(set(new.array))

del new, hotels_df



#%%

#Load the model

checkpoint = 'epoch_2_features_model_save_sample2000'

#Load Electra - Model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
model = AutoModelForSequenceClassification.from_pretrained(wd+'\\Models\\'+checkpoint)

#%%

#Get all features

all_reviews=[]

for file in ['All_reviews.csv','PostPandemic_reviews.csv','PrePandemic_reviews.csv']:
    #Extract only the top 100 features for each list
    all_reviews.append(pd.read_csv(wd+'\\Data\\Tfidf\\'+file, index_col=0).iloc[0:100, :])

all_reviews = pd.concat(all_reviews, ignore_index=True)

#Get the unique values, so we don't repeat
all_features = all_reviews.Features.unique()


#%%

#Here is the one I'm working on.
#This one actually makes sense


#progress_bar = tqdm(range(sentiment_grid.shape[0]*sentiment_grid.shape[1]))

big_collect=[]

for feature in tqdm(all_features):
    for pand_status in ['before', 'after']:
        for state in states:
            for i in range(sample_sentiment.shape[0]):
                
                word = sample_sentiment.iloc[i].Word
                sentiment = sample_sentiment.iloc[i].Sentiment
                
                sample_sentence = 'The '+feature.strip()+' was '+word.strip()+'.'+' This hotel is in '+state+'.'\
                    + ' I stayed '+ pand_status + ' the pandemic.'
        
                big_collect.append([sample_sentence, feature, word, sentiment, state, pand_status])
            

big_collect = pd.DataFrame(big_collect, columns = ['Sentence', 'Feature', 'Word', 'Sentiment', 'State', 'Pandemic_Timing'])
big_collect.to_csv(wd+'\\Data\\Features\\Sentences\\feature_sentence_matrix.csv', index=True)



#%%

def model_electa_feature_extract(save_name, checkpoint_in):
    """
    Run the accuracy test  - how well does the model perfom on the test data
    Prints the accuracy score, also returns pred and true values.

    Parameters
    ----------
    save_name : str
        the name of the save directory for the model - will also load the test data.

    Returns
    -------
    pred
        the predicted values for the test set.
    true
        the true values for the test set.

    """

    #Load Basic Electra - Model and tokenizer
    checkpoint = "google/electra-base-discriminator"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(wd+'//models//'+checkpoint_in)


    #Load Data
    dataset = load_dataset('csv', data_files={'test':wd+'\\Data\\Cleaned\\Split\\sm_sampled_data_train_features.csv'})
    #dataset = load_dataset('csv', data_files={'test': wd+'\\Data\\Features\\Sentences\\feature_sentence_matrix.csv'})

    
    ###  Tokenize  ###
    def tokenize_function(example):
        return tokenizer(example["Review_Body"],truncation=True)
    
    
    #Prepare the data
    dataset = dataset.rename_column("Sentence", "Review_Body")
    #dataset = dataset.rename_column("Unnamed: 0", "labels")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    #Remove raw text - already tokenized and masked
    tokenized_datasets = tokenized_datasets.remove_columns(["Review_Body", "Unnamed: 0"])
    
    #Note: Removing all non-text columns?
    tokenized_datasets = tokenized_datasets.remove_columns(['Feature', 'Word', 'Sentiment', 'State', 'Pandemic_Timing'])
    

    
    #Rename and reformat columns
    #tokenized_datasets = tokenized_datasets.rename_column("Review_rating", "labels")
    tokenized_datasets.set_format("torch", columns=tokenized_datasets["test"].column_names)
    tokenized_datasets["test"].column_names
    
    device = torch.device("cpu")
    model.to(device)
    device
    
    ###  Dataloader  ###

    eval_dataloader = DataLoader(tokenized_datasets["test"],
                                 batch_size=8, collate_fn=data_collator)
    
    
    for batch in eval_dataloader:
        break
    
    print('here')
    ###  Set-up  ###
    num_training_steps = len(eval_dataloader)
    progress_bar = tqdm(range(num_training_steps))
    
    
    pred = []
    index_val = []
    
    ###  Evaluations  ###
    
    model.eval()
    for batch in eval_dataloader:
        progress_bar.update(1)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            
    
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        print(predictions.cpu().numpy())

        pred.append(predictions.cpu().numpy())
        #index_val.append(batch["labels"])
    


    return pred

#%%


def model_ernie_feature_extract(save_name, checkpoint_in):
    """
    Run the accuracy test  - how well does the model perfom on the test data
    Prints the accuracy score, also returns pred and true values.

    Parameters
    ----------
    save_name : str
        the name of the save directory for the model - will also load the test data.

    Returns
    -------
    pred
        the predicted values for the test set.
    true
        the true values for the test set.

    """

    #Load Basic Electra - Model and tokenizer
    checkpoint = "nghuyong/ernie-2.0-en"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(wd+'//models//'+checkpoint_in)


    #Load Data
    #dataset = load_dataset('csv', data_files={'test':wd+'\\Data\\Cleaned\\Split\\sm_sampled_data_train_features.csv'})
    dataset = load_dataset('csv', data_files={'test': wd+'\\Data\\Features\\Sentences\\feature_sentence_matrix.csv'})


    ###  Tokenize  ###
    def tokenize_function(example):
        return tokenizer(example["Review_Body"],truncation=True,max_length=512, padding=True,)
    
    
    #Prepare the data
    dataset = dataset.rename_column("Sentence", "Review_Body")
    #dataset = dataset.rename_column("Unnamed: 0", "labels")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    #Remove raw text - already tokenized and masked
    tokenized_datasets = tokenized_datasets.remove_columns(["Review_Body", "Unnamed: 0"])
    
    #Note: Removing all non-text columns?
    tokenized_datasets = tokenized_datasets.remove_columns(['Feature', 'Word', 'Sentiment', 'State', 'Pandemic_Timing'])
    

    
    #Rename and reformat columns
    #tokenized_datasets = tokenized_datasets.rename_column("Review_rating", "labels")
    tokenized_datasets.set_format("torch", columns=tokenized_datasets["test"].column_names)
    tokenized_datasets["test"].column_names
    
    device = torch.device("cuda")
    model.to(device)
    device
    
    ###  Dataloader  ###

    eval_dataloader = DataLoader(tokenized_datasets["test"],
                                 batch_size=5, collate_fn=data_collator)
    
    
    for batch in eval_dataloader:
        break
    
    print('here')
    ###  Set-up  ###
    num_training_steps = len(eval_dataloader)
    progress_bar = tqdm(range(num_training_steps))
    
    
    pred = []
    index_val = []
    
    ###  Evaluations  ###
    
    model.eval()
    for batch in eval_dataloader:
        progress_bar.update(1)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            
    
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        print(predictions.cpu().numpy())

        pred.append(predictions.cpu().numpy())
        #index_val.append(batch["labels"])
    


    return pred
#%%

# Generate the predicted values
#Note: Do this on gpu, very slow on cpu

predictions = model_electa_feature_extract(save_name='test_sample_1', checkpoint_in='epoch_0_features_model_save_sample2000_5epoch')

import csv 
with open(wd+'\\Data\\Features\\Results\\predictions.csv', 'w') as f: 
    write = csv.writer(f) 
    write.writerows(predictions) 
    
    
#%%
predictions = model_ernie_feature_extract(save_name='test_sample_ernie', checkpoint_in='test_baselines_reordered_pandemic_state')

import csv 
with open(wd+'\\Data\\Features\\Results\\ernie_feature_predictions.csv', 'w') as f: 
    write = csv.writer(f) 
    write.writerows(predictions) 
    


#%%

##############
#
#  Load in the features and predictions
#
##############


features = pd.read_csv(wd+'\\Data\\Features\\Sentences\\feature_sentence_matrix.csv', index_col=0)
predictions = pd.read_csv(wd+'\\Data\\Features\\Results\\predictions.csv',header=None)


#Clean the predictions
pred_melt = predictions.T.melt()
pred_melt = pred_melt.drop('variable', axis=1)
pred_melt = pred_melt.rename({'value':'Prediction'}, axis=1)
pred_melt['Prediction'] = pred_melt['Prediction']+1

#Join features and predictions
features = features.join(pred_melt)



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






