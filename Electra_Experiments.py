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

# Import the city's Hotels and Reviews


###
# Load in all Hotels
###

#Loop through all cities - Collect all Hotels
all_files = os.listdir(wd+'\\Data\\Cleaned\\')
all_cities = [file for file in all_files if '.pkl' in file and 'Hotels' in file]

#Start with just one
all_cities = [all_cities[1]]

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

#Rejoin to the dataframe
hotels_df['State'] = new


###
# Load in all Reviews
###

#Loop through all cities - Collect all Hotels
all_files = os.listdir(wd+'\\Data\\Cleaned\\')
all_cities = [file for file in all_files if '.pkl' in file and 'Reviews' in file]

#Start with just one
all_cities = [all_cities[1]]

hotels = []
for city in tqdm(all_cities):
    hotels_file =  pd.read_pickle(wd+'\\Data\\Cleaned\\'+city)
    
    #Drop some columns to make the dataframe lighter
    hotels_file.drop(['Review_date', 'Reviewer_loc','Review_stay_date', 'tokens','tokens_joined',
                       'Review_Year', 'Review_Month','Stay_Year', 'Stay_Month'], 
                     axis=1, inplace=True)
    
    #Merge the title and body, then drop the individuals
    hotels_file['Review_Body'] = hotels_file['Review_title']+'. '+hotels_file['Review_text'] 
    hotels_file.drop(['Review_title', 'Review_text'], 
                     axis=1, inplace=True)    
    
    #Add to the list
    hotels.append(hotels_file)


#Merge into one dataframe
reviews_df = pd.concat(hotels)
reviews_df.rename({'Hotel_ID':'hotel_ID'}, axis=1, inplace=True)

#Delete intermediaries to free up memory
del all_files, all_cities, city, hotels, hotels_file


## Merge the states to the Reviews


#Get the states 
state_df = hotels_df.filter(['hotel_ID', 'State']).copy()

#Merge the states into each review, for grouping.
reviews_df = reviews_df.merge(state_df, on = 'hotel_ID')

#Drop the hotel ID - not needed
reviews_df.drop(['hotel_ID'],axis=1, inplace=True)  



#Get a sample, because you canot open the full dataset
sample = reviews_df.sample(n=2000)



#%%

##Save the sample as a test for the Electra Model

sample['Review_rating'] = sample['Review_rating'] -1


train, test = train_test_split(sample, test_size=0.4, random_state=42)
test, validation = train_test_split(test, test_size=0.5, random_state=42)

train.to_csv(wd+'\\Data\\Cleaned\\Split\\train_sample.csv', index=False)
test.to_csv(wd+'\\Data\\Cleaned\\Split\\test_sample.csv', index=False)
validation.to_csv(wd+'\\Data\\Cleaned\\Split\\validation_sample.csv', index=False)

#%%

################################
#
#  Modeling with ELECTRA
#
################################


#Load Electra - Model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
model = AutoModelForSequenceClassification.from_pretrained("google/electra-base-discriminator")


#Load Data
dataset = load_dataset('csv', data_files={'train': wd+'\\Data\\Cleaned\\Split\\train_sample.csv',
                                          'test': wd+'\\Data\\Cleaned\\Split\\test_sample.csv',
                                          'validation':wd+'\\Data\\Cleaned\\Split\\validation_sample.csv'})


###  Tokenize  ###
def tokenize_function(example):
    return tokenizer(example["Review_Body"],truncation=True)


#Prepare the data
tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#Remove raw text - already tokenized and masked
tokenized_datasets = tokenized_datasets.remove_columns(["Review_Body"])

#Note: Removing all non-text columns?
tokenized_datasets = tokenized_datasets.remove_columns(['State',"Review_PrePandemic", 'Stay_PrePandemic'])

#Rename and reformat columns
tokenized_datasets = tokenized_datasets.rename_column("Review_rating", "labels")
tokenized_datasets.set_format("torch", columns=tokenized_datasets["train"].column_names)
tokenized_datasets["train"].column_names



###  Dataloader  ###
train_dataloader = DataLoader(tokenized_datasets["train"],
                              shuffle=True, batch_size=8, collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_datasets["validation"],
                             batch_size=8, collate_fn=data_collator)


for batch in train_dataloader:
    break

#%%



###  Set-up  ###
num_epochs = 8
checkpoint = "google/electra-base-discriminator"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)



###  Model Set-up  ###
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=5)
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)
optimizer = AdamW(model.parameters(), lr=5e-5)

num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler( "linear", optimizer=optimizer,
                              num_warmup_steps=0, num_training_steps=num_training_steps)
print(num_training_steps)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
device

#Tqdm
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

###  Evaluations  ###
metric = load_metric("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()






























