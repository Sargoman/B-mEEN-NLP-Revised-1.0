# -*- coding: utf-8 -*-
"""
Created on Tue May  3 02:19:10 2022

@author: SAURAV GUPTA
"""
import re
import os
import pandas as pd

path = "D:/Career/Company AI_ML assignments/UST/UST Global/project/Sample data - BioNLP-ST_2013_CG_sample-1.0"
file_dict = {}
for filename in os.listdir(path):
    with open(os.path.join(path,filename), 'r') as f:
        text = f.read()
        file_dict[filename] = text
        #print(text)
    #print(filename)
    
df_file = pd.DataFrame(columns=['files','elements'])
df_file['files'] = file_dict.keys()
df_file['elements'] = file_dict.values()
#df_file

for i in range(len(df_file['files'])):
    df_file['files'][i] = df_file['files'][i].split('.')
#df_file['files'][2][1]

#temp_data = df_file['elements'][0]

def format_elements(data):
    ls_data = data.split('\n')
    ls_list_data = [ls_data[i] for i in range(len(ls_data)) if len(ls_data[i])>0]
    ls_ls_data = [re.sub("\t",",",i).split(",") for i in ls_list_data]
    chemical_data_list = [i for i in ls_ls_data if len(i) == 3]
    return chemical_data_list
#temp_dataX = format_elements(temp_data)

def ent_position(data):
    temp_dataXX = data[1].split(' ')
    temp_ls = []
    temp_ls = temp_dataXX[0]
    temp_dataXX[0] = int(temp_dataXX[1])
    temp_dataXX[1] = int(temp_dataXX[2])
    temp_dataXX[2] = temp_ls
    return temp_dataXX   
#ent_position(temp_dataX[0])

list_of_elements = []
for i in range(len(df_file['files'])):
    if df_file['files'][i][1] != 'txt':
        list_of_elements.append(format_elements(df_file['elements'][i]))
    else:
        list_of_elements.append('')  

df_file['list_of_elements'] = list_of_elements
#df_file['elements'][3]
#len(df_file['list_of_elements'][1][1])
position_ent_ls = []
for i in range(len(df_file['list_of_elements'])):
    temp_list = []
    if len(df_file['list_of_elements'][i]) != 0:
        for j in range(len(df_file['list_of_elements'][i])):
            temp_list.append(ent_position(df_file['list_of_elements'][i][j]))
    else:
        temp_list.append(df_file['elements'][i])
    position_ent_ls.append(temp_list)
            
df_file['training_test_data_elements'] = position_ent_ls

#len(df_file['training_test_data_elements'][0] + df_file['training_test_data_elements'][2])
#len(position_ent_ls[0] + position_ent_ls[1])
#len(df_file['training_test_data_elements'][0])
all_ent_doc = []
clinical_string = []
for i in range(len(df_file['files'])-1):
    if len(df_file['training_test_data_elements'][i+1]) != 1 and df_file['files'][i][0] == df_file['files'][i+1][0]:
        all_ent_doc.append(df_file['training_test_data_elements'][i] + df_file['training_test_data_elements'][i+1])
    if len(df_file['training_test_data_elements'][i]) == 1:
        clinical_string.append(df_file['training_test_data_elements'][i])
clinical_string.append(df_file['training_test_data_elements'][29])        

clinical_string_ls_tup = []
for i in range(len(all_ent_doc)):
    temp_clinical_string_tup = []
    for j in range(len(all_ent_doc[i])):
        temp_clinical_string_tup.append(tuple(map(lambda x:x, all_ent_doc[i][j])))
    clinical_string_ls_tup.append(temp_clinical_string_tup)

#type(clinical_string[0][0])54
#type(clinical_string_ls_tup[0])
x_format = [] 
for i in range(len(all_ent_doc)):
    temp_dict = {}
    temp_dict["entities"] = clinical_string_ls_tup[i]
    temp_val = clinical_string[i][0],temp_dict
    x_format.append(temp_val)
#print(x_format[0])


###########################################################################
trainData = x_format[0:8]
devData = x_format[8:10]

import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
nlp = spacy.blank('en') # load a new spacy model
db = DocBin() # create a DocBin object
for text, annot in tqdm(trainData): # data in previous format
    doc = nlp.make_doc(text) # create doc object from text
    ents = []
    for start, end, label in annot['entities']: # add character indexes
        span = doc.char_span(start, end, label=label, alignment_mode='contract')
        if span is None:
            print('Skipping entity')
        else:
            ents.append(span)
    try:
        doc.ents = ents # label the text with the ents
        db.add(doc)
    except:
        print(text, annot)
db.to_disk('./train.spacy') # save the docbin object


for text, annot in tqdm(devData): # data in previous format
    doc = nlp.make_doc(text) # create doc object from text
    ents = []
    for start, end, label in annot['entities']: # add character indexes
        span = doc.char_span(start, end, label=label, alignment_mode='contract')
        if span is None:
            print('Skipping entity')
        else:
            ents.append(span)
    try:
        doc.ents = ents # label the text with the ents
        db.add(doc)
    except:
        print(text, annot)
db.to_disk('./dev.spacy')


