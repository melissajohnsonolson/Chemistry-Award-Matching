# -*- coding: utf-8 -*-
"""
Created on Fri May  3 17:30:19 2019

@author: johns
"""
import os
import pandas as pd
import numpy as np
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import re
import datetime
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import matplotlib.pyplot as plt
import string

os.chdir('C:/Users/johns/Documents/Machine Learning/science funding project')



def most_similar(new_text):
    plt.style.use('ggplot')
    #Load the trained model
    model = Doc2Vec.load('doc2vec_abstracts')    
    
    #Load the awards data
    awds = pd.read_csv('NSF CHE 2015.csv', encoding='latin-1')
    awds['StartDate'] = pd.to_datetime(awds['StartDate']).apply(lambda x: x.year)
    awds['EndDate'] = pd.to_datetime(awds['EndDate'])
    awds['AwardedAmountToDate']=[x.replace('$', '') for x in awds['AwardedAmountToDate']]
    awds['AwardedAmountToDate']=[x.replace(',', '') for x in awds['AwardedAmountToDate']]
    awds['AwardedAmountToDate']=pd.to_numeric(awds['AwardedAmountToDate'])
    
    #Load the papers data sheet
    papers = pd.read_csv('che_paper_data.csv')
    papers['year'] = pd.to_datetime(papers['year'])
    papers['citations per year'] = papers['citations'].divide(
        [((datetime.datetime.today()-x).days)/365.2422 for x in papers['year']])  
    papers['year'] = papers['year'].apply(lambda x:x.year)

    #Here we build up and instantiate the stop words and lemmatizer  
    stop    = set(stopwords.words('english'))
    exclude = set(string.punctuation) 
    lemma   = WordNetLemmatizer()
    boiler_plate = 'This award reflects NSF''s statutory mission and has been deemed worthy of support through evaluation using the Foundation''s intellectual merit and broader impacts review criteria'    
    
    #The function below cleans and tokenizes the input text
    def word_mod(doc):
        doc = re.sub('<.*?>', ' ', doc)
        doc = re.sub(boiler_plate, '', doc)
        punct_free  = ''.join(ch for ch in doc if ch not in exclude)
        words   = punct_free.lower().split()
        stop_free  = " ".join([i for i in words if i not in stop])
        lemm = " ".join(lemma.lemmatize(word) for word in stop_free.split())
        word_list = lemm.split()
        # only take words which are greater than 2 characters
        cleaned = [word for word in word_list if len(word) > 2]
        return cleaned 
    #Here the cleaned up text is fed to the model. The model returns the similiarty of this text to all awards
    #We print out the two most similar award numbers
    new_text_clean = model.infer_vector(word_mod(new_text))
    sims = model.docvecs.most_similar([new_text_clean], topn=len(model.docvecs))
    sim1 = sims[0]
    sim2 = sims[1]
    print('The most similar award numbers are {0} and {1}, with similarity scores of {2} and {3}.'.format(sim1[0], sim2[0], round(sim1[1],3), round(sim2[1],3)))
    
    #Here we examine the awards with similarity score greater than 0.5. It matches
    #with other awards made, the amount of the award, and the publication data
    #from each award.
    
    sims = [sims[i][0] for i in range(len(sims)) if sims[i][1]>0.5]
    sim_awards = awds[awds['AwardNumber'].isin(sims)].copy()
    sim_papers = papers[papers['award number'].isin(sims)].copy()
    
    #Here plots for different data and metrics are generated.
    fig1 = plt.figure()
    sim_awards.groupby('StartDate')['AwardNumber'].count().plot.bar(rot = 0)
    plt.title('Awards per Year Similar to Text')
    plt.ylabel('Number of Awards')
    plt.xlabel('Year of Award')
    plt.show()
   
    fig2 = plt.figure()
    sim_awards.groupby('StartDate')['AwardedAmountToDate'].sum().plot.bar(rot = 0)
    plt.title('Total Awarded Dollars per Year for Awards Similar to Text')
    plt.ylabel('Total Dollars Awarded')
    plt.xlabel('Year of Award')
    plt.show()
   
    fig3 = plt.figure()
    sim_papers.groupby('year')['title'].count().plot.bar(rot = 0)
    plt.title('Number of Publications Each Year from Awards Similar to Text')
    plt.ylabel('Number of Publications')
    plt.xlabel('Year of Publication')
    plt.show()
   
    fig4 = plt.figure()
    sim_papers.boxplot(column = ['citations per year'], by =  'year')
    plt.title('Citations per Year For \n Publications from Awards Similar to Text')
    plt.suptitle("")
    plt.ylabel('Citations per Year')
    plt.xlabel('Year of Publication')
    plt.show()
    
most_similar(new_text)