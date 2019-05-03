# -*- coding: utf-8 -*-
"""
Created on Fri May  3 17:28:24 2019

@author: johns
"""

import os
import pandas as pd
import numpy as np
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import re
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import string

os.chdir('C:/Users/johns/Documents/Machine Learning/science funding project')
awds = pd.read_csv('NSF CHE 2012.csv', encoding='latin-1')
papers = pd.read_csv('che_paper_data.csv')
pgms = ['6878', '6880', '6882', '6883', '6884', '6885', '9101', '9102', '6881']

def doc2vec_abstracts(awds):
    #First we need to filter the data by program code. Some grants have multiple program
    #codes, so we first filter through to determine which cells contain the program code
    #then we replace the exisiting program code(s) with the provided one. This ensures there
    #is only one code per award.

    pgms = ['6878', '6880', '6882', '6883', '6884', '6885', '9101', '9102', '6881']
    
    awds = awds[awds['ProgramElementCode(s)'].str.contains('|'.join(pgms))]
    for x in pgms:
        awds['ProgramElementCode(s)'] = np.where(awds['ProgramElementCode(s)'].str.contains(x), x, awds['ProgramElementCode(s)'] )
   
    abstracts = awds[['AwardNumber','Abstract']].copy()
    #This is a pretty clean data set, but there are some empty entries, so we
    #filter them out here
    abstracts = abstracts.dropna()
    
    
    #Here we start building our dictinary and creating the cleaned up corpus.
    #We start by  removing stop words, punctuation, and stemming or lemmatizing
    #he abstract text
    stop    = set(stopwords.words('english'))
    exclude = set(string.punctuation) 
    lemma   = WordNetLemmatizer()
    boiler_plate = 'This award reflects NSF''s statutory mission and has been deemed worthy of support through evaluation using the Foundation''s intellectual merit and broader impacts review criteria' 
  
    
    #This function applies the bigram and trigram functions and lemmatizes the 
    #the abstracts and only keeps words that a greater than 2 characters
    def word_mod(doc):
        doc = re.sub('<.*?>', ' ', doc)
        doc = re.sub(boiler_plate, '', doc)
        doc = re.sub('-', ' ', doc)
        punct_free  = ''.join(ch for ch in doc if ch not in exclude)
        words   = punct_free.lower().split()
        stop_free  = " ".join([i for i in words if i not in stop])
        lemm = " ".join(lemma.lemmatize(word) for word in stop_free.split())
        word_list = lemm.split()
        # only take words which are greater than 2 characters
        cleaned = [word for word in word_list if len(word) > 2]
        return cleaned
    
    abstracts['clean_abstracts'] = [word_mod(doc) for doc in abstracts['Abstract']] 
    
   
    train_corpus = [gensim.models.doc2vec.TaggedDocument(abstracts['clean_abstracts'].iloc[i], [str(abstracts['AwardNumber'].iloc[i])]) for i in range(len(abstracts))]
    
    model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=100, dm=0)    
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    
    rank = 0
    sim =[]
    for i in range(len(abstracts)):
        inferred_vector = model.infer_vector(abstracts['clean_abstracts'].iloc[i])
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        if abstracts['AwardNumber'].iloc[i] != int(sims[0][0]):
            rank = rank +1
        else:
            sim.append(sims[0][1])
    simscore = pd.Series(sim).mean()
    percent = (1-rank/len(train_corpus))*100
    print('The model correctly matched the training set {0}% of the time with an average similiarity of {1}'.format(round(percent,1) ,round(simscore,3)))
    
    model.save('doc2vec_abstracts')
    
   
    
doc2vec_abstracts(awds)  