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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


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

#Rejoin to the dataframe
hotels_df['State'] = new


###
# Load in all Reviews
###

#Loop through all cities - Collect all Hotels
all_files = os.listdir(wd+'\\Data\\Cleaned\\')
all_cities = [file for file in all_files if '.pkl' in file and 'Reviews' in file]

#Start with just one
#all_cities = [all_cities[1]]

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
state_df = hotels_df.filter(['hotel_ID', 'State', 'hotel_location_walk']).copy()

#Merge the states into each review, for grouping.
reviews_df = reviews_df.merge(state_df, on = 'hotel_ID')

#Drop the hotel ID - not needed
reviews_df.drop(['hotel_ID'],axis=1, inplace=True)  

#%%


################################
#
#  Subsample the data 
#  Equal samples from each state
#  Sample size: 8000
#
################################

collect_sampler = []


for state in tqdm(reviews_df.State.unique()):
    subset_state = (reviews_df[reviews_df.State == state]).sample(n=60, replace=False, random_state=42)
    collect_sampler.append(subset_state)



collect_sampler = pd.concat(collect_sampler)

#Adjust so the ratings start at 0
collect_sampler['Review_rating'] = collect_sampler['Review_rating'] -1

#Split into train, test and validation
train, test = train_test_split(collect_sampler, test_size=0.4, random_state=42)
test, validation = train_test_split(test, test_size=0.5, random_state=42)


#%%

#Get a sample, because you canot open the full dataset
sample = reviews_df.sample(n=2000, random_state=42)


##Save the sample as a test for the Electra Model
# 60/20/20 Split

sample['Review_rating'] = sample['Review_rating'] -1


train, test = train_test_split(sample, test_size=0.4, random_state=42)
test, validation = train_test_split(test, test_size=0.5, random_state=42)


#%%



## Check how it would look if only predict 4

train.Review_rating.plot.hist()

from sklearn.metrics import accuracy_score


# Calculate the F1 score.
f1 = accuracy_score(y_true=test["Review_rating"], y_pred=[4]*test.shape[0])

print('Accurancy if you always guess the most frequent rating (4)')
print('Accuracy: %.4f' % f1)

#Accuracy f1: 0.4930
#Accuracy f1_sampled: 0.4739

#%%


def baseline_model_electa(train_in, val_in, test_in, save_name):


    #Load Basic Electra - Model and tokenizer
    checkpoint = "google/electra-base-discriminator"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    #Save Data
    train_in.to_csv(wd+'\\Data\\Cleaned\\Split\\baseline_train_sample.csv', index=False)
    test_in.to_csv(wd+'\\Data\\Cleaned\\Split\\baseline_test_sample_'+save_name+'.csv', index=False)
    val_in.to_csv(wd+'\\Data\\Cleaned\\Split\\baseline_validation_sample.csv', index=False)
    
    del train_in, test_in, val_in
    
    
    #Load Data
    dataset = load_dataset('csv', data_files={'train': wd+'\\Data\\Cleaned\\Split\\baseline_train_sample.csv',
                                              'validation':wd+'\\Data\\Cleaned\\Split\\baseline_validation_sample.csv'})
    
    
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
    
    try:
        tokenized_datasets = tokenized_datasets.remove_columns(['hotel_location_walk'])
    except:
        pass
    
    try:
        tokenized_datasets = tokenized_datasets.remove_columns(['location'])
    except:
        pass
    
    #Rename and reformat columns
    tokenized_datasets = tokenized_datasets.rename_column("Review_rating", "labels")
    tokenized_datasets.set_format("torch", columns=tokenized_datasets["train"].column_names)
    tokenized_datasets["train"].column_names
    
    
    
    ###  Dataloader  ###
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True,
                                 batch_size=8, collate_fn=data_collator)
    eval_dataloader = DataLoader(tokenized_datasets["validation"],
                                 batch_size=8, collate_fn=data_collator)
    
    
    for batch in train_dataloader:
        break
    
    
    ###  Set-up  ###
    num_epochs = 3
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=5)
    
    
    ###  Model Set-up  ###
    outputs = model(**batch)
    print(outputs.loss, outputs.logits.shape)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler( "linear", optimizer=optimizer,
                                  num_warmup_steps=0, num_training_steps=num_training_steps)
    print(num_training_steps)
    
    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
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
        
        #Save at the end of each epoch
        #model.save_pretrained(wd+'//models//epoch_'+str(epoch)+'_features_model_save')
        
    
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
    
    print('Validation Accuracy:')
    print(metric.compute())
    model.save_pretrained(wd+'//models//baseline_'+save_name+'_model_save')
    
    return model




def baseline_model_electa_test(save_name):
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
    model = AutoModelForSequenceClassification.from_pretrained(wd+'//models//baseline_'+save_name+'_model_save')


    #Load Data
    dataset = load_dataset('csv', data_files={'test': wd+'\\Data\\Cleaned\\Split\\baseline_test_sample_'+save_name+'.csv'})
    
    
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
    
    try:
        tokenized_datasets = tokenized_datasets.remove_columns(['hotel_location_walk'])
    except:
        pass
    
    try:
        tokenized_datasets = tokenized_datasets.remove_columns(['location'])
    except:
        pass
    
    #Rename and reformat columns
    tokenized_datasets = tokenized_datasets.rename_column("Review_rating", "labels")
    tokenized_datasets.set_format("torch", columns=tokenized_datasets["test"].column_names)
    tokenized_datasets["test"].column_names
    
    device = torch.device("cpu")
    model.to(device)
    device
    
    ###  Dataloader  ###

    eval_dataloader = DataLoader(tokenized_datasets["test"],
                                 batch_size=20, collate_fn=data_collator)
    
    
    for batch in eval_dataloader:
        break
    
    print('here')
    ###  Set-up  ###
    num_training_steps = len(eval_dataloader)
    progress_bar = tqdm(range(num_training_steps))
    
    
    pred = []
    true = []
    
    ###  Evaluations  ###
    metric = load_metric("accuracy")
    model.eval()
    for batch in eval_dataloader:
        progress_bar.update(1)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
    
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
        pred.append(predictions.cpu().numpy())
        true.append(batch["labels"].cpu().numpy())
    
    print('Validation Accuracy:')
    print(metric.compute())

    return pred, true


#%%

# Basic baseline - just text

model_name = 'basic_s'

#Train model with only the base text
basic_model = baseline_model_electa(save_name = model_name, train_in=train, val_in=validation, test_in=test)

#Get the baseline accuracy
baseline_metric, baseline_true = baseline_model_electa_test(save_name = model_name)
#Results:0.720 Accuracy

print(f'\n\n{model_name} Predictions\n--------------\n')
print([metric.numpy() for metric in baseline_metric])


#%%

#Add Pandemic Timing

def add_features_pandemic(df_in):
    
    df_in = df_in.copy()
    #Filter for review_prepandemic col
    boolean_filter = {True:'before', False:'after'}
    
    #Create additional text to add to the review
    new_text = '.'+ ' I stayed '+ df_in.Review_PrePandemic.map(boolean_filter) + ' the pandemic.'

    df_in.Review_Body = df_in.Review_Body + new_text
    
    return df_in


train_mod = add_features_pandemic(train)
test_mod = add_features_pandemic(test)
val_mod = add_features_pandemic(validation)
print('\n\nFeatures Added\n\n')

model_name = 'pandemic_s'
pandemic_model = baseline_model_electa(save_name = model_name, train_in=train, val_in=validation, test_in=test)

print('\n\nTraining Complete\n\n')

#Get the baseline accuracy
baseline_metric, baseline_true = baseline_model_electa_test(save_name = model_name)
#Results:  Accuracy

print(f'\n\n{model_name} Predictions\n--------------\n')
print([metric.numpy() for metric in baseline_metric])

#%%

#Add Pandemic Timing and State

def add_features_pandemic_state(df_in):
    
    df_in = df_in.copy()
    #Filter for review_prepandemic col
    boolean_filter = {True:'before', False:'after'}
    
    #Create additional text to add to the review
    new_text = '.'+' This hotel is in '+df_in.State+'.'\
    + ' I stayed '+ df_in.Review_PrePandemic.map(boolean_filter) + ' the pandemic.'

    df_in.Review_Body = df_in.Review_Body + new_text
    
    return df_in


train_mod = add_features_pandemic_state(train)
test_mod = add_features_pandemic_state(test)
val_mod = add_features_pandemic_state(validation)
print('\n\nFeatures Added\n\n')

model_name = 'pandemic_state_s'
pandemic_model = baseline_model_electa(save_name = model_name, train_in=train, val_in=validation, test_in=test)

print('\n\nTraining Complete\n\n')

#Get the baseline accuracy
baseline_metric, baseline_true = baseline_model_electa_test(save_name = model_name)
#Results:  Accuracy

print(f'\n\n{model_name} Predictions\n--------------\n')
print([metric.numpy() for metric in baseline_metric])


#%%

#Add Pandemic Timing and State and Walkability


def walkability_str_gen(num):
    if num <= 50:
        
        if num > 25:
            walk_str= 'not very walkable.'
        else:
            walk_str= 'not walkable at all.'
            
    else: #Num score 50+
        if num > 75: #most walkable
            walk_str= 'very walkable.'
        else:
            walk_str= 'fairly walkable.'
    return walk_str

def add_features_walk(df_in):
    
    df_in = df_in.copy()
    #Filter for review_prepandemic col
    boolean_filter = {True:'before', False:'after'}
    
    #Create additional text to add to the review
    new_text = '.'+' This hotel is in '+df_in.State+'.'\
    + ' I stayed '+ df_in.Review_PrePandemic.map(boolean_filter) + ' the pandemic.'\
    + ' This hotel was ' + df_in.hotel_location_walk.map(walkability_str_gen)

    df_in.Review_Body = df_in.Review_Body + new_text
    
    return df_in


train_mod = add_features_walk(train)
test_mod = add_features_walk(test)
val_mod = add_features_walk(validation)
print('\n\nFeatures Added\n\n')

model_name = 'pandemic_state_walk_s'
pandemic_model = baseline_model_electa(save_name = model_name, train_in=train, val_in=validation, test_in=test)

print('\n\nTraining Complete\n\n')

#Get the baseline accuracy
baseline_metric, baseline_true = baseline_model_electa_test(save_name = model_name)
#Results:  Accuracy

print(f'\n\n{model_name} Predictions\n--------------\n')
print([metric.numpy() for metric in baseline_metric])

#%%


def walkability_str_gen(num):
    if num <= 50:
        
        if num > 10:
            walk_str= 'not very walkable.'
        else:
            walk_str= 'not walkable at all.'
            
    else: #Num score 50+
        if num > 80: #most walkable
            walk_str= 'very walkable.'
        else:
            walk_str= 'fairly walkable.'
    return walk_str

def add_features_walk(df_in):
    
    df_in = df_in.copy()
    #Filter for review_prepandemic col
    boolean_filter = {True:'before', False:'after'}
    
    #Create additional text to add to the review
    new_text = '.'+ ' This hotel was ' + df_in.hotel_location_walk.map(walkability_str_gen)

    df_in.Review_Body = df_in.Review_Body + new_text
    
    return df_in


train_mod = add_features_walk(train)
test_mod = add_features_walk(test)
val_mod = add_features_walk(validation)
print('\n\nFeatures Added\n\n')

model_name = 'walk_only_s'
pandemic_model = baseline_model_electa(save_name = model_name, train_in=train, val_in=validation, test_in=test)

print('\n\nTraining Complete\n\n')

#Get the baseline accuracy
baseline_metric, baseline_true = baseline_model_electa_test(save_name = model_name)
#Results:  Accuracy

print(f'\n\n{model_name} Predictions\n--------------\n')
print([metric.numpy() for metric in baseline_metric])


#%%

#Add just State


def add_features_state(df_in):
    
    df_in = df_in.copy()

    
    #Create additional text to add to the review
    new_text = '.'+' This hotel is in '+df_in.State+'.'
    
    df_in.Review_Body = df_in.Review_Body + new_text
    
    return df_in


train_mod = add_features_state(train)
test_mod = add_features_state(test)
val_mod = add_features_state(validation)
print('\n\nFeatures Added\n\n')

model_name = 'state_only'
pandemic_model = baseline_model_electa(save_name = model_name, train_in=train, val_in=validation, test_in=test)

print('\n\nTraining Complete\n\n')

#Get the baseline accuracy
baseline_metric, baseline_true = baseline_model_electa_test(save_name = model_name)
#Results:  Accuracy

print(f'\n\n{model_name} Predictions\n--------------\n')
print([metric.numpy() for metric in baseline_metric])

#%%


#Add Pandemic Timing and State and Walkability


def walkability_str_gen(num):
    if num <= 50:
        
        if num > 10:
            walk_str= 'not very walkable.'
        else:
            walk_str= 'not walkable at all.'
            
    else: #Num score 50+
        if num > 80: #most walkable
            walk_str= 'very walkable.'
        else:
            walk_str= 'fairly walkable.'
    return walk_str

def add_features__state_walk(df_in):
    
    df_in = df_in.copy()
    #Filter for review_prepandemic col
    boolean_filter = {True:'before', False:'after'}
    
    #Create additional text to add to the review
    new_text = '.'+' This hotel is in '+df_in.State+'.'\
    + ' This hotel was ' + df_in.hotel_location_walk.map(walkability_str_gen)

    df_in.Review_Body = df_in.Review_Body + new_text
    
    return df_in


train_mod = add_features__state_walk(train)
test_mod = add_features__state_walk(test)
val_mod = add_features__state_walk(validation)
print('\n\nFeatures Added\n\n')

model_name = 'just_state_walk'
pandemic_model = baseline_model_electa(save_name = model_name, train_in=train, val_in=validation, test_in=test)

print('\n\nTraining Complete\n\n')

#Get the baseline accuracy
baseline_metric, baseline_true = baseline_model_electa_test(save_name = model_name)
#Results:  Accuracy

print(f'\n\n{model_name} Predictions\n--------------\n')
print([metric.numpy() for metric in baseline_metric])


#%%




















#%%
#
# If we want to do more stats
#
#


#Load Electra - Model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")




sample_sentence = test['Review_Body'].iloc[2]


input_ids = torch.tensor(tokenizer.encode(sample_sentence, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
scores = basic_model(input_ids)[0]
print(scores)

#Print the predicted score
prediction = torch.argmax(model(input_ids).logits, dim=-1)
print(prediction)



basic_model(test['Review_Body'].iloc[0])





from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

print 'Accuracy:', accuracy_score(y_test, prediction)
print 'F1 score:', f1_score(y_test, prediction)
print 'Recall:', recall_score(y_test, prediction)
print 'Precision:', precision_score(y_test, prediction)
print '\n clasification report:\n', classification_report(y_test,prediction)
print '\n confussion matrix:\n',confusion_matrix(y_test, prediction)





#%%

test.to_csv(wd+'\\Data\\Cleaned\\Split\\untrained_test_sample.csv', index=False)

#Load Basic Electra - Model and tokenizer
checkpoint = "google/electra-base-discriminator"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=5)


#Load Data
dataset = load_dataset('csv', data_files={'test': wd+'\\Data\\Cleaned\\Split\\untrained_test_sample.csv'})


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

try:
    tokenized_datasets = tokenized_datasets.remove_columns(['hotel_location_walk'])
except:
    pass

try:
    tokenized_datasets = tokenized_datasets.remove_columns(['location'])
except:
    pass

#Rename and reformat columns
tokenized_datasets = tokenized_datasets.rename_column("Review_rating", "labels")
tokenized_datasets.set_format("torch", columns=tokenized_datasets["test"].column_names)
tokenized_datasets["test"].column_names

device = torch.device("cpu")
model.to(device)
device

###  Dataloader  ###

eval_dataloader = DataLoader(tokenized_datasets["test"],
                             batch_size=20, collate_fn=data_collator)


for batch in eval_dataloader:
    break

print('here')
###  Set-up  ###
num_training_steps = len(eval_dataloader)
progress_bar = tqdm(range(num_training_steps))


pred = []
true = []

###  Evaluations  ###
metric = load_metric("accuracy")
model.eval()
for batch in eval_dataloader:
    progress_bar.update(1)
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])
    pred.append(predictions)
    true.append(batch["labels"])

print('Validation Accuracy:')
print(metric.compute())

#%%

#######
#
# Batch testing the baselines for MSE
#
#######

def model_output_cleaner(data_in, col_name):
    
    predictions = pd.DataFrame(data_in)
    pred_melt = predictions.T.melt()
    pred_melt = pred_melt.drop('variable', axis=1)
    pred_melt = pred_melt.rename({'value':col_name}, axis=1)
    pred_melt[col_name] = pred_melt[col_name]+1

    return pred_melt


#Running all baselines
collector_mse = {}
collector_acc = {}
problems = []

#for model_name in ['basic_s', 'pandemic_s','pandemic','pandemic_state_s', 'pandemic_state_walk_s','walk_only_s','state_only', 'state_only_s','just_state_walk','just_state_walk_s',]:   
for model_name in ['pandemic','pandemic_state_s', 'pandemic_state_walk_s','walk_only_s','state_only', 'state_only_s','just_state_walk','just_state_walk_s',]:
    
    try:
        baseline_metric, baseline_true = baseline_model_electa_test(save_name = model_name)
        
        base_metric = model_output_cleaner(baseline_metric, 'Baseline').dropna(axis=0)
        base_true = model_output_cleaner(baseline_true, 'True').dropna(axis=0)
        
        mse_score = mean_squared_error(base_metric, base_true)
        collector_mse[model_name] = mse_score
        
        acc_score = accuracy_score(base_metric, base_true)
        collector_acc[model_name] = acc_score
        
        print(f'\n\n------------------\n\n {model_name} MSE:{mse_score}\n {model_name} ACC:{acc_score}\n\n--------------\n')
    except:
        problems.append(model_name)
    
        print(f'\n\n------------------\n\n {model_name} is a problem! \n\n--------------\n')

print(problems)
#%%













