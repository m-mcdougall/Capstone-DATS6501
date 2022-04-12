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

sample_pos = collector_pos.sample(n=20, replace=False, random_state=42)
sample_neg =collector_neg.sample(n=20, replace=False, random_state=42)

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

#Get all features

pand_only_reviews=[]

for file in ['PostPandemic_reviews.csv','PrePandemic_reviews.csv']:
    #Extract only the top 100 features for each list
    pand_only_reviews.append(pd.read_csv(wd+'\\Data\\Tfidf\\'+file, index_col=0).iloc[0:100, :])

pand_only_reviews = pd.concat(pand_only_reviews, ignore_index=True)

#Get the unique values, so we don't repeat
pand_only_reviews = pand_only_reviews.Features.unique()



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

for item in pand_only_reviews:
    if item not in all_features:
        print(item)

# These are the same - So no need to use All_reviews.csv.

#%%

#Here is the one I'm working on.
#This one actually makes sense


#progress_bar = tqdm(range(sentiment_grid.shape[0]*sentiment_grid.shape[1]))

big_collect=[]

for feature in tqdm(all_features):
    for pand_status in ['Before', 'After']:
        for state in states:
            for walkability in range(4):
                for i in range(sample_sentiment.shape[0]):
                    
                    word = sample_sentiment.iloc[i].Word
                    sentiment = sample_sentiment.iloc[i].Sentiment
                    
                    sample_sentence = 'State '+ state + '. '+ pand_status + ' pandemic.'\
                        +'Walkability '+ str(walkability) +'. '\
                        +feature.strip().capitalize()+' '+word.strip().lower()+'.'
                    
            
                    big_collect.append([sample_sentence, feature, word, sentiment, state, pand_status])
                

big_collect = pd.DataFrame(big_collect, columns = ['Sentence', 'Feature', 'Word', 'Sentiment', 'State', 'Pandemic_Timing'])
big_collect.to_csv(wd+'\\Data\\Features\\Sentences\\reorder_feature_sentence_matrix.csv', index=True)



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