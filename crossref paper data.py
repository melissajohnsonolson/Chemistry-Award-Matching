# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 09:19:45 2019

@author: johns
"""
import datetime, os
from habanero import Crossref
import pandas as pd
os.chdir('C:/Users/johns/Documents/Machine Learning/science funding project')


cr = Crossref()
awds=pd.read_csv('NSF CHE 2012.csv', encoding='latin-1')
paper_data_base = pd.DataFrame(columns = ['award number','title','year', 'citations', 'type', 'publication', 'DOI',
                                          'funders','authors'])

for y in range(0,len(awds)):
    award_number = awds.loc[y]['AwardNumber']
    awd_search_list = 'CHE-'+str(award_number)
    
    search = cr.works(filter = {'award_number': awd_search_list}, limit = 200)
    
    for x in range(0,len(search['message']['items'])):
        title = search['message']['items'][x]['title'][0]
        type1 = search['message']['items'][x]['type']
        referenced_by = search['message']['items'][x]['is-referenced-by-count']
        year1 = datetime.datetime.strptime(search['message']['items'][x]['created']['date-time'], "%Y-%m-%dT%H:%M:%SZ")
        publication = search['message']['items'][x]['container-title'][0]
        DOI = search['message']['items'][x]['DOI']
        funders = search['message']['items'][x]['funder']
        author_list =  search['message']['items'][x]['author']
     
        paper_data_base.loc[len(paper_data_base) + 1] = [award_number, title, year1, referenced_by, type1, publication,
                            DOI, funders, author_list]

paper_data_base.to_csv('che_paper_data.csv', index = False)