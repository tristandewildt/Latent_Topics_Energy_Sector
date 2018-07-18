'''
Created on 8 dec. 2016

@author: tewdewildt
'''
from __future__ import division
import gensim
from gensim import corpora, models, similarities
from pprint import pprint
import pickle
from six import iteritems
from math import *
import logging
import pyLDAvis.gensim
import winsound
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from itertools import repeat
import copy
import operator
from operator import itemgetter
from scipy.spatial.distance import pdist, squareform
from gensim.models.ldamodel import LdaModel



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

version = 'long'

threshold_calc = 1

dict_values_in_articles = False
dict_co_values_in_articles = False
create_array_freq_co_values = False
print_graph_freq_co_values = False
study_co_values = False
get_topics_of_articles = False
get_topics_of_co_values = False
create_distribution_topics_values = False
study_topics = False
set_topics_to_values = True
study_topics_to_values = False

calculate_distribution_topics_in_values = False
find_articles_co_values_per_topic = False
create_network_data = False
create_Hellinger_distance_graph = False
get_most_cited_articles_in_topics = False
get_articles_most_corresponding_to_topics = False
get_most_cited_articles_in_topics_per_value = False
get_all_values_topics = False
values_per_topic = False


if version == 'short':
    
    efficiency = ['effectiveness', 'efficacy', 'ineffectiveness', 'inefficiency', 'productivity', 'performance', 'efficiency', 'efficient']
    compabitility = ['interoperability', 'standards', 'harmonization', 'standardization', 'interchangeability', 'replacable', 'replaceability', 'consistency', 'incompatibility', 'protocol', 'platform', 'interface', 'compabitility']
    intelligence = ['sense', 'smart', 'learning', 'learn', 'machine-learning', 'reasoning', 'artificial intelligence', 'Intelligence', 'sensor']
    robustness = ['fitness', 'resilience', 'strength', 'unbreakable', 'adaptability', 'integrity', 'breakable', 'collapse', 'failure', 'reliability', 'maintanability', 'resiliency', 'robustness']
    safety_health = ['safeness', 'distress', 'endangerment', 'imperilment', 'jeopardy', 'peril', 'healthiness', 'illness', 'sickness', 'unhealthiness', 'dreadful', 'hazard', 'wellbeing', 'safe', 'harmful', 'health']
    environmental_sustainability = ['unsustainable', 'sustainability', 'sustainable', 'ecological', 'eco-friendly', 'nature-friendly', 'environmentally-friendly', 'intergenerational', 'renewable', 'environmental', 'climate', 'sustainability']
    justice = ['equity', 'fair', 'inequity', 'injustice', 'impartial', 'unfair', 'unbiased', 'justice', 'objectivity', 'lawful', 'egalitarian', 'distributive']
    privacy = ['hack', 'hacker', 'cybersecurity', 'cyber', 'internet of things', 'data protection', 'privacy']
    competitiveness = ['competitor', 'contestant', 'rival', 'noncompetitor', 'market structure', 'barriers to entry', 'monopoly', 'oligopoly', 'competition', 'contestability', 'strategic behavior', 'competition', 'complementary assets', 'competitive advantage', 'stakeholders', 'competitiveness']
    innovativeness = ['creativeness', 'innovativeness', 'ingeniousness', 'ingenuity', 'creativity', 'invention', 'inventiveness', 'innovation', 'innovate']
    #no_value = []
    #economic_development = ['assets', 'capital', 'cost-effective', 'cost-effectiveness', 'business model', 'profit', 'affordable', 'cost', 'market', 'welfare', 'wealth', 'monetary', 'price', 'tariff', 'financial', 'prosperity', 'monetary valuation', 'economic growth', 'merit order', 'poverty', 'revenue', 'debt'] 

else:
    efficiency = ['effective', 'ineffective', 'effectiveness', 'ineffectiveness', 'efficacy', 'inefficacy', 'efficacious', 'inefficacious', 'efficiency', 'inefficiency', 'efficient', 'inefficient', 'productivity', 'improductivity', 'productiveness', 'improductiveness', 'performance', 'performances', 'nonperformance', 'nonperformances', 'non performance', 'non performances', 'non-performance', 'non-performances']
    compabitility = ['interoperability', 'standards', 'standardization', 'standardizations', 'standardisation', 'standardisations', 'harmonisation', 'harmonisations', 'interchangeable', 'interchangeability', 'interchangeabilities', 'replacable', 'replacability', 'replacabilities', 'consistency', 'consistencies', 'incompatibility', 'incompatibilities', 'protocol', 'protocols', 'platform', 'platforms', 'interface', 'interfaces']
    intelligence = ['senses', 'sense', 'smart', 'learn', 'learning', 'machine-learning', 'reasoning', 'reasonings', 'artificial intelligence', 'Intelligence', 'sensor']
    robustness = ['fitness', 'fitnesses', 'resilient', 'irresilient', 'resilience', 'resiliences', 'strength', 'strengths', 'unbreakable', 'unbreakables', 'adaptable', 'adaptability', 'unadaptable', 'inadaptability', 'integrity', 'breakable', 'breakability', 'collapse', 'collapses', 'failure', 'failures', 'reliable', 'unreliable', 'reliability', 'unreliability', 'maintainable', 'maintanability', 'robust', 'robustness']
    safety_health = ['safe', 'unsafe', 'safety', 'unsafety', 'safeness', 'distress', 'distresses', 'endangerment', 'endangerments', 'imperilment', 'imperilments', 'jeopardy', 'peril', 'perils', 'health', 'healthy', 'unhealthy', 'healthiness', 'unhealthiness', 'ill', 'illness', 'illnesses', 'sick', 'sickness', 'sicknesses', 'dreadful', 'dreadfulness', 'hazard', 'hazards', 'wellbeing', 'wellbeings', 'harmful', 'harmfulness']
    environmental_sustainability = ['sustainable', 'unsustainable', 'sustainability', 'ecological', 'eco friendly', 'eco-friendly', 'eco unfriendly', 'eco-unfriendly', 'environmentally friendly', 'environmentally-friendly', 'environmentally unfriendly', 'environmentally-unfriendly', 'nature friendly', 'nature-friendly', 'nature unfriendly', 'nature-unfriendly', 'intergenerational', 'renewable', 'renewables', 'climate', 'climates']
    #justice = ['justice']
    justice = ['equity', 'inequity', 'fair', 'unfair', 'justice', 'injustice', 'impartial', 'unbiased', 'objectivity', 'lawful', 'unlawful', 'egalitarian', 'inegalitarian', 'distributive', 'fairness', 'justness', 'impartiality', 'equitable']
    privacy = ['hack', 'hacks', 'hacker', 'hackers', 'cybersecurity', 'cybersecurities', 'cyber security', 'cyber securities', 'cyber-security', 'cyber-securities', 'cyber', 'internet of things', 'data protection', 'data-protection', 'privacy']
    competitiveness = ['competitor', 'contestant', 'rival', 'noncompetitor', 'market structure', 'barriers to entry', 'monopoly', 'oligopoly', 'competition', 'contestability', 'strategic behavior', 'competition', 'complementary assets', 'competitive advantage', 'stakeholders', 'competitiveness']
    innovativeness = ['creativeness', 'innovativeness', 'ingeniousness', 'ingenuity', 'creativity', 'invention', 'inventiveness', 'innovation', 'innovate']
    #economic_development = ['asset', 'assets', 'capital', 'capitals', 'cost effective', 'cost ineffective', 'cost-effective', 'cost-ineffective', 'cost effectiveness', 'cost effectiveness', 'cost-effectiveness', 'cost-effectiveness', 'business model', 'business models', 'profit', 'profits', 'affordable', 'cost', 'costs', 'market', 'markets', 'welfare', 'wealth', 'monetary', 'monetary', 'wealth', 'price', 'prices', 'tariff', 'tariffs', 'financial', 'prosperity', 'prosperities', 'monetary valuation', 'monetary valuations', 'monetary-valuation', 'monetary-valuations', 'economic growth', 'economic growths', 'economic-growth', 'economic-growths', 'poverty', 'poverties', 'poverty', 'poverties', 'revenue', 'debt', 'debts']

values_names = ['efficiency', 'compabitility', 'intelligence', 'robustness', 'safety_health', 'environmental_sustainability', 'justice', 'privacy', 'competitiveness', 'innovativeness'] 

values = [efficiency, compabitility, intelligence, robustness, safety_health, environmental_sustainability, justice, privacy, competitiveness, innovativeness]


if dict_values_in_articles == True:
    print('Working on dict_values_in_articles...')
    with open ('../../Backup/Paper analysis/3. Retrieval_documents_energy_sector/Output_4_topics/0.33333/scopus_txt_topics_energy(4)_onethird', 'rb') as fp:
        scopus_list_txt = pickle.load(fp)
    #with open ('../../Backup/Paper analysis/3. Retrieval_documents_energy_sector/Output_4_topics/0.33333/scopus_txt_topics_energy(4)_DOI_onethird', 'rb') as fp:
    #    scopus_list_DOI = pickle.load(fp)
    
    itr1 = 0
    dict_articles_values = {}
    for i in range(len(scopus_list_txt)):
        itr2 = 0
        values_in_articles = {}
        for h in values:
            #f = sum(c in h for c in scopus_list_txt[i]) # Score is here the sum; we might also consider a percentage or something in between
            words_of_value_in_article = {}
            itr3 = 0
            for c in h:
                #print(c)
                f = scopus_list_txt[i].count(c)  
                #f =  sum(c in scopus_list_txt[i])
                words_of_value_in_article[itr3] = f
                itr3 += 1
            values_in_articles[itr2] = words_of_value_in_article
            itr2 += 1          
        itr1 += 1
        dict_articles_values[itr1] = values_in_articles
        #if itr1 % 1000 == 0:
        #    print(str(itr1)+" docs treated out of "+str(len(scopus_list_txt)))
    
    with open('../Save/'+str(version)+'_'+str(threshold_calc)+'/scopus_list_txt_energy(4)_articles_to_values','wb') as fp:
        pickle.dump(dict_articles_values, fp)

if dict_co_values_in_articles == True:
    
    print('Working on dict_co_values_in_articles...')
    with open ('../Save/'+str(version)+'_'+str(threshold_calc)+'/scopus_list_txt_energy(4)_articles_to_values', 'rb') as fp:
        dict_articles_values = pickle.load(fp)
    
    dict_articles_co_values = {}
    for f in range(len(dict_articles_values)):
        n = 3
        array_conflicting_values = [[[0 for k in xrange(3)] for j in xrange(len(values))] for i in xrange(len(values))]
        x = 0
        for i in range(len(values_names)):
            y = 0
            for g in range(len(values_names)):
                #score = sqrt(dict_articles_values[f+1][i] * dict_articles_values[f+1][g])
                #score = sqrt(pow(dict_articles_values[f+1][i],2) * pow(dict_articles_values[f+1][g],2))
                            
                if (sum(v for v in dict_articles_values[f+1][i].values() if v > 0) >= threshold_calc) and (sum(v for v in dict_articles_values[f+1][g].values() if v > 0) >= threshold_calc):
                    score = sqrt(pow(sum(v for v in dict_articles_values[f+1][i].values() if v > 0),2) * pow(sum(v for v in dict_articles_values[f+1][g].values() if v > 0),2))
                else:
                    score = 0
                
                
                #if (dict_articles_values[f+1][i] >= threshold_calc) and (dict_articles_values[f+1][g] >= threshold_calc):
                #    score = sqrt(pow(dict_articles_values[f+1][i],2) * pow(dict_articles_values[f+1][g],2))
                #else:
                #    score = 0
           
                array_conflicting_values[x][y][0]= x
                array_conflicting_values[x][y][1]= y
                array_conflicting_values[x][y][2]= score
                y += 1
            x += 1
       
        dict_articles_co_values[f] = array_conflicting_values
    #print(dict_articles_co_values)
          
    with open('../Save/'+str(version)+'_'+str(threshold_calc)+'/scopus_list_txt_energy(4)_articles_to_co_values','wb') as fp:
        pickle.dump(dict_articles_co_values, fp)

if create_array_freq_co_values == True:
    print('Working on create_array_freq_co_values...')
    with open ('../Save/'+str(version)+'_'+str(threshold_calc)+'/scopus_list_txt_energy(4)_articles_to_co_values', 'rb') as fp:
        dict_articles_co_values = pickle.load(fp)
        
    n = 3
    array_freq_co_values = [[[0 for k in xrange(3)] for j in xrange(len(values))] for i in xrange(len(values))]
    
    for x in range(len(values)):
        for y in range(len(values)):
            score_freq_co_values = 0
            for f in range(len(dict_articles_co_values)):  
                #score_freq_co_values = score_freq_co_values + dict_articles_co_values[f][x][y][2]
                #print(dict_articles_co_values[f][x][y][2])
                ru = 0
                if dict_articles_co_values[f][x][y][2] > 0:
                    ru = 1
                score_freq_co_values = score_freq_co_values + ru
                
            array_freq_co_values[x][y][0]= x
            array_freq_co_values[x][y][1]= y
            array_freq_co_values[x][y][2]= score_freq_co_values
    
    print(array_freq_co_values)
            
    with open('../Save/'+str(version)+'_'+str(threshold_calc)+'/scopus_list_txt_energy(4)_array_freq_co_values','wb') as fp:
        pickle.dump(array_freq_co_values, fp)
    
if print_graph_freq_co_values == True:
    print('Working on print_graph_freq_co_values...')
    with open ('../Save/'+str(version)+'_'+str(threshold_calc)+'/scopus_list_txt_energy(4)_array_freq_co_values', 'rb') as fp:
        array_freq_co_values = pickle.load(fp)

    op = 0
    for i in range(len(values)):
        ip = 0
        for g in range(len(values)):

            if ip >= op:
                array_freq_co_values[i][g][2]= 0
            ip += 1 
        op += 1     
    
    x = []
    r = 0
    for i in range(len(values)):
        x = x + ([r] * len(values))
        r +=1
    y = []
    for i in range(len(values)):
        y = y + range(len(values))
       
    colors = 'red'
    area = []
    for i in range(len(values)):
        for t in range(len(values)):
            area.append(array_freq_co_values[i][t][2] * 0.5)
                       
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=area, c=colors, alpha=0.5)
    ax.yaxis.tick_right()
        
    xy = []
    for t in range(len(x)):
        z = [x[t], y[t]]
        z = np.array(z)
        xy.append(z)
        
    Values_index = range(len(values))
    plt.xticks(Values_index, values_names, rotation = 90)
    plt.yticks(Values_index, values_names)
    plt.grid()
    plt.show()

if study_co_values == True:
    print('Working on study_co_values...')
    with open ('../Save/'+str(version)+'_'+str(threshold_calc)+'/scopus_list_txt_energy(4)_articles_to_co_values', 'rb') as fp:
        dict_articles_co_values = pickle.load(fp)
    #with open ('../../Backup/Paper analysis/3. Retrieval_documents_energy_sector/Output_4_topics/0.33333/scopus_txt_topics_energy(4)_onethird', 'rb') as fp:
    #    scopus_list_txt = pickle.load(fp)
    with open ('../../Backup/Paper analysis/3. Retrieval_documents_energy_sector/Output_4_topics/0.33333/scopus_txt_topics_energy(4)_DOI_onethird', 'rb') as fp:
        scopus_list_DOI = pickle.load(fp)

#    print(dict_articles_co_values[0])
#    print(dict_articles_co_values[1])
#    print(dict_articles_co_values[2])
#    print(len(dict_articles_co_values))
    
    for value_1 in range(len(values_names)):
        
        
        for value_2 in range(len(values_names)):
    
    
            dict_speci_co_values = {}
            for i in range(len(dict_articles_co_values)):
                dict_speci_co_values[i] = dict_articles_co_values[i][value_1][value_2][2]
                #print(dict_speci_co_values[i])
                #if (value_1 == 4) and (value_2 == 6) and dict_articles_co_values[i][value_1][value_2][2] > 0:
                #    print(dict_articles_co_values[i][value_1][value_2][2])
                
            dict_speci_co_values = sorted(dict_speci_co_values.items(), key=lambda x:x[1], reverse = True)    #dict_speci_co_values_reduced = [t for t in dict_speci_co_values_reduced if t[1] >= 1]
            dict_speci_co_values_reduced_threshold = filter(lambda t: t[1] > threshold_calc, dict_speci_co_values)
            #print(dict_speci_co_values_reduced_threshold)
            dict_speci_co_values_reduced_top_100 = dict_speci_co_values[0:100]
            
            
            #print(len(dict_speci_co_values_reduced_threshold))
            
            list_DOI_co_values_reduced_top_100 = []
            for i in range(len(dict_speci_co_values_reduced_top_100)):
                list_DOI_co_values_reduced_top_100.append(scopus_list_DOI[dict_speci_co_values_reduced_top_100[i][0]])
            #print(list_DOI_co_values_reduced_top_100)
            
            
            #print(len(dict_speci_co_values_reduced_threshold))
            #print(dict_speci_co_values_reduced_threshold[1])
            #print(dict_speci_co_values_reduced_threshold[2])
            #print(dict_speci_co_values_reduced_threshold[3])
            
            with open('../Save/'+str(version)+'_'+str(threshold_calc)+'/scopus_list_txt_energy(4)_threshold_'+str(threshold_calc)+'_'+str(value_1)+'_'+str(value_2),'wb') as fp:
                pickle.dump(dict_speci_co_values_reduced_threshold, fp)
            
            with open('../Save/'+str(version)+'_'+str(threshold_calc)+'/scopus_list_txt_energy(4)_top_100_'+str(value_1)+'_'+str(value_2),'wb') as fp:
                pickle.dump(dict_speci_co_values_reduced_top_100, fp)
                
            save_text_DOI_list = True
            if save_text_DOI_list == True:
                #text_list = []
                DOI_list = []
                for i in range(len(dict_speci_co_values_reduced_threshold)):
                    #text_list.append(scopus_list_txt[dict_speci_co_values_reduced_threshold[i][0]])
                    DOI_list.append(scopus_list_DOI[dict_speci_co_values_reduced_threshold[i][0]])
                
                #print(DOI_list)
                
                #with open('../Save/'+str(version)+'_'+str(threshold_calc)+'/text_list_value_threshold_'+str(threshold_calc)+'_'+str(value_1)+'_and_'+str(value_2),'wb') as fp:
                #    pickle.dump(text_list, fp)
                with open('../Save/'+str(version)+'_'+str(threshold_calc)+'/DOI_list_value_threshold_'+str(threshold_calc)+'_'+str(value_1)+'_and_'+str(value_2),'wb') as fp:
                    pickle.dump(DOI_list, fp)


if get_topics_of_co_values == True:

    num_topics = 100
    
    value_1 = 1
    value_2 = 2
    topics = [49]
    
    threshold_topic = 0.25
    
    with open ('../Save/'+str(version)+'_'+str(threshold_calc)+'/DOI_list_value_threshold_'+str(threshold_calc)+'_'+str(value_1)+'_and_'+str(value_2), 'rb') as fp:
        DOI_list = pickle.load(fp)
        
    with open ('../../Backup/Paper analysis/3. Retrieval_documents_energy_sector/Output_4_topics/0.33333/scopus_txt_topics_energy(4)_DOI_onethird', 'rb') as fp:
        scopus_list_DOI = pickle.load(fp)
        
    #with open ('../../Backup/Paper analysis/3. Retrieval_documents_energy_sector/Output_4_topics/0.33333/scopus_txt_topics_energy(4)_onethird', 'rb') as fp:
    #    scopus_list_txt = pickle.load(fp)
    
    lda = models.LdaModel.load('../../Backup/Paper analysis/5. Coherence_tests_energy/4_topics/modelLDA'+str(num_topics)+'_topics_energy(4)_onethird_coh.lda')
    corpus_tfidf = corpora.MmCorpus('../../Backup/Paper analysis/5. Coherence_tests_energy/4_topics/scopus_corpus'+str(num_topics)+' topics_energy(4)_onethird_coh.mm')
    dictionary = corpora.Dictionary.load('../../Backup/Paper analysis/5. Coherence_tests_energy/4_topics/scopus_list'+str(num_topics)+'_energy(4)_onethird_coh.dict')

    all_topics = lda.get_document_topics(corpus_tfidf, minimum_probability=0, per_word_topics = False)
    #print(lda.show_topics(num_topics= num_topics, num_words=10, log=True, formatted=True))
        
    dict_co_values_articles_topics = {}
    for i in range(num_topics):
        dict_co_values_articles_topics[i]=[]
        #print(dict_co_values_articles_topics)
    
    for i in DOI_list:
        if not not i:
            for h in all_topics[scopus_list_DOI.index(i)]:
                if h[1] >= threshold_topic:
                    dict_co_values_articles_topics[h[0]].append(i)
                    
    
    for i in dict_co_values_articles_topics:
        x = len(dict_co_values_articles_topics[i])
        dict_co_values_articles_topics[i].insert(0, x)
        #print(dict_co_values_articles_topics)
        
    #pprint(dict_co_values_articles_topics)          

    for i in topics:
        ty = str(dict_co_values_articles_topics[i]).replace("', '", ") OR DOI(")
        ty = ty.replace(" '", " DOI(")
        ty = ty.replace("'", ")")
        print(str(i)+': '+str(ty))
        #print(ty)
        #print(str(i)+': '+str(dict_co_values_articles_topics[i])) 


if create_distribution_topics_values == True:

    num_topics = 100
    threshold_topic = 0.33
    

           
    with open ('../../Backup/Paper analysis/3. Retrieval_documents_energy_sector/Output_4_topics/0.33333/scopus_txt_topics_energy(4)_DOI_onethird', 'rb') as fp:
        scopus_list_DOI = pickle.load(fp)
    
    
    lda = models.LdaModel.load('../../Backup/Paper analysis/5. Coherence_tests_energy/4_topics/modelLDA'+str(num_topics)+'_topics_energy(4)_onethird_coh.lda')
    corpus_tfidf = corpora.MmCorpus('../../Backup/Paper analysis/5. Coherence_tests_energy/4_topics/scopus_corpus'+str(num_topics)+' topics_energy(4)_onethird_coh.mm')
    dictionary = corpora.Dictionary.load('../../Backup/Paper analysis/5. Coherence_tests_energy/4_topics/scopus_list'+str(num_topics)+'_energy(4)_onethird_coh.dict')
    
    all_topics = lda.get_document_topics(corpus_tfidf, minimum_probability=0, per_word_topics = False)
        #print(lda.show_topics(num_topics= num_topics, num_words=10, log=True, formatted=True))
        
    dict_values_in_topics = {}
    for i in range(len(values)):
        dict_values_in_topics[i]=[]
    #print(dict_values_in_topics)
    
    
    dict_co_values_articles_topics = {}
    for i in range(num_topics):
        dict_co_values_articles_topics[i]=[]
        dict_co_values_articles_topics[i]=dict_values_in_topics
        
    #print(dict_co_values_articles_topics)
    
    for y in range(len(values)):
        
        value_1 = y
        value_2 = y
    
        with open ('../Save/'+str(version)+'_'+str(threshold_calc)+'/DOI_list_value_threshold_'+str(threshold_calc)+'_'+str(value_1)+'_and_'+str(value_2), 'rb') as fp:
            DOI_list = pickle.load(fp)

        for i in DOI_list:
            #if not not i:
            for h in all_topics[scopus_list_DOI.index(i)]:
                if h[1] >= threshold_topic:
                    dict_co_values_articles_topics[h[0]][value_1].append(i)
        print(y)
        #print(dict_co_values_articles_topics)               

    with open('../Save/'+str(version)+'_'+str(threshold_calc)+'/DOI_list_value_threshold_'+str(threshold_calc)+'_only_'+str(value_1),'wb') as fp:
        pickle.dump(dict_co_values_articles_topics, fp)


if study_topics == True:
    
    num_topics = 100
    threshold_topic = 0
           
    lda = models.LdaModel.load('../../Backup/Paper analysis/5. Coherence_tests_energy/4_topics/modelLDA'+str(num_topics)+'_topics_energy(4)_onethird_coh.lda')
    corpus_tfidf = corpora.MmCorpus('../../Backup/Paper analysis/5. Coherence_tests_energy/4_topics/scopus_corpus'+str(num_topics)+' topics_energy(4)_onethird_coh.mm')
    dictionary = corpora.Dictionary.load('../../Backup/Paper analysis/5. Coherence_tests_energy/4_topics/scopus_list'+str(num_topics)+'_energy(4)_onethird_coh.dict')
    
    all_topics = lda.get_document_topics(corpus_tfidf, minimum_probability=0, per_word_topics = False)
        #print(lda.show_topics(num_topics= num_topics, num_words=10, log=True, formatted=True))

    k = 0
    topics_of_articles = {}
    for o in all_topics:
        v = 0
        listy = []
        for h in o:
            if o[v][1] >= threshold_topic:
                listy.append(o[v][0])
            v += 1
        topics_of_articles[k] = listy      
            
        #print(topics_of_articles)
        if k % 1000 == 0:        
            print(k)
        k += 1      

    with open('../Save/topics_of_articles_'+str(threshold_topic)+'_'+str(num_topics)+'_topics','wb') as fp:
        pickle.dump(topics_of_articles, fp)

if set_topics_to_values == True:

    num_topics = 100
    threshold_topic = 0.25
    
    with open ('../../Backup/Paper analysis/3. Retrieval_documents_energy_sector/Output_4_topics/0.33333/scopus_txt_topics_energy(4)_DOI_onethird', 'rb') as fp:
        scopus_list_DOI = pickle.load(fp)
        
    print(len(scopus_list_DOI))
    
    with open ('../../Backup/Paper analysis/3. Retrieval_documents_energy_sector/Output_4_topics/0.33333/scopus_list_topics_energy(4)_citation_counts_onethird', 'rb') as fp:
        scopus_list_cit_counts = pickle.load(fp)
    
    with open ('../Save/topics_of_articles_'+str(threshold_topic)+'_'+str(num_topics)+'_topics', 'rb') as fp:
        topics_of_articles = pickle.load(fp)

    with open ('../../Backup/Paper analysis/3. Retrieval_documents_energy_sector/Output_4_topics/0.33333/scopus_txt_topics_energy(4)_onethird', 'rb') as fp:
        scopus_list_txt = pickle.load(fp)
    
    dict_values_in_topics = {}
    counta = 0
    
    
    
    for v in values:
        topics_in_value = {}
        for vy in range(num_topics):
            topics_in_value[vy] = []
        
        for h in range(len(scopus_list_txt)):
            listy = [i for i in v if i in scopus_list_txt[h]]
            if len(listy) > 0:
                if not not scopus_list_DOI[h]:
                    for i in topics_of_articles[h]:
                        lista = [scopus_list_DOI[h], scopus_list_cit_counts[h]]
                        topics_in_value[i].append(lista)
                        #print(topics_in_value)
                        
        dict_values_in_topics[counta] = topics_in_value
        print(counta)
        print(dict_values_in_topics)
        counta += 1         
                
    with open('../Save/topics_per_value_'+str(threshold_topic)+'_'+str(num_topics)+'_topics','wb') as fp:
        pickle.dump(dict_values_in_topics, fp)           
        
if study_topics_to_values == True:
    
    num_topics = 100
    threshold_topic = 0.25
    value_researched = 6
    min_frequency = 0
    #topics_with_marg_distr_higher_0_9 = [96, 84, 14, 42, 32, 68, 49, 25, 63, 41, 92, 67, 10, 39, 38, 79, 47, 7, 31, 99, 46, 70, 93, 97, 71, 4, 73, 18, 35, 15, 82, 66, 19, 37, 9, 72, 8, 94]
    topics_with_marg_distr_higher_0_9 = range(0,99,1)
    
    with open ('../Save/topics_per_value_'+str(threshold_topic)+'_'+str(num_topics)+'_topics', 'rb') as fp:
        dict_values_in_topics = pickle.load(fp)
    
    with open ('../Save/topics_of_articles_'+str(threshold_topic)+'_'+str(num_topics)+'_topics', 'rb') as fp:
        topics_of_articles = pickle.load(fp)
    
    lda = models.LdaModel.load('../../Backup/Paper analysis/5. Coherence_tests_energy/4_topics/modelLDA'+str(num_topics)+'_topics_energy(4)_onethird_coh.lda')
    corpus_tfidf = corpora.MmCorpus('../../Backup/Paper analysis/5. Coherence_tests_energy/4_topics/scopus_corpus'+str(num_topics)+' topics_energy(4)_onethird_coh.mm')
    dictionary = corpora.Dictionary.load('../../Backup/Paper analysis/5. Coherence_tests_energy/4_topics/scopus_list'+str(num_topics)+'_energy(4)_onethird_coh.dict')
    
    dict_count_freq_topics = {}
    for x in range(num_topics):
        dict_count_freq_topics[x]=0
    
    for key, value in topics_of_articles.items():
        for i in value:
            dict_count_freq_topics[i] = dict_count_freq_topics[i] + 1
    #print(dict_count_freq_topics)       

    
    #print(dict_values_in_topics[value_researched])
    
    #for key, value in dict_values_in_topics[value_researched].items():
    #    print(key, len(value))

    dict_freq_value_in_topic = {}
    for x in range(num_topics):
        if x in topics_with_marg_distr_higher_0_9:
        
        
        #if dict_count_freq_topics[x] >= min_frequency:
            dict_freq_value_in_topic[x] = len(dict_values_in_topics[value_researched][x]) / dict_count_freq_topics[x]
    print(dict_freq_value_in_topic)

    tuple_sorted_dict = sorted(dict_freq_value_in_topic.items(), key=operator.itemgetter(1), reverse=True)
    print(tuple_sorted_dict)
    print(len(tuple_sorted_dict))
    
    
    county = 0
    for i in tuple_sorted_dict:
        print(str(i[0])+': '+str(lda.print_topic(i[0])))
        county += 1
        if county >= 100:
            break



if calculate_distribution_topics_in_values == True:
    
    num_topics = 100
    threshold_topic = 0.33
    min_frequency = 0
    topics_with_marg_distr_higher_0_9 = [96, 84, 14, 42, 32, 68, 49, 25, 63, 41, 92, 67, 10, 39, 38, 79, 47, 7, 31, 99, 46, 70, 93, 97, 71, 4, 73, 18, 35, 15, 82, 66, 19, 37, 9, 72, 8, 94]
    
    
    with open ('../Save/topics_per_value_'+str(threshold_topic)+'_'+str(num_topics)+'_topics', 'rb') as fp:
        dict_values_in_topics = pickle.load(fp)
    
    #print(dict_values_in_topics[6][97])
    #print(len(dict_values_in_topics[6][97]))
    
    
    
    with open ('../Save/topics_of_articles_'+str(threshold_topic)+'_'+str(num_topics)+'_topics', 'rb') as fp:
        topics_of_articles = pickle.load(fp)
    
    lda = models.LdaModel.load('../../Backup/Paper analysis/5. Coherence_tests_energy/4_topics/modelLDA'+str(num_topics)+'_topics_energy(4)_onethird_coh.lda')
    corpus_tfidf = corpora.MmCorpus('../../Backup/Paper analysis/5. Coherence_tests_energy/4_topics/scopus_corpus'+str(num_topics)+' topics_energy(4)_onethird_coh.mm')
    dictionary = corpora.Dictionary.load('../../Backup/Paper analysis/5. Coherence_tests_energy/4_topics/scopus_list'+str(num_topics)+'_energy(4)_onethird_coh.dict')
    
    dict_count_freq_topics = {}
    for x in range(num_topics):
        dict_count_freq_topics[x]=0
    
    for key, value in topics_of_articles.items():
        for i in value:
            dict_count_freq_topics[i] = dict_count_freq_topics[i] + 1

    dict_freq_values_in_topics = {}
    for op in range(len(values)):
        
        dict_freq_value_in_topic = {}
        
        count_yes = 0
        count_no = 0
        
        average_yes = 0
        average_no = 0
        
        for x in range(num_topics):
            #print(x)
            
            if x in topics_with_marg_distr_higher_0_9:
                count_yes += 1
                dict_freq_value_in_topic[x] = len(dict_values_in_topics[op][x]) / dict_count_freq_topics[x]
                average_yes = average_yes + dict_count_freq_topics[x]
                
                #print(dict_count_freq_topics[x])
            else:
                count_no += 1
                average_no = average_no + dict_count_freq_topics[x]
                
                #print(x)
                #print(dict_count_freq_topics[x])
                #print("")
                
            
            
            
            #if dict_count_freq_topics[x] >= min_frequency:
            #    count_yes += 1
            #    dict_freq_value_in_topic[x] = len(dict_values_in_topics[op][x]) / dict_count_freq_topics[x]
            #else:
            #    count_no += 1
        #print(dict_freq_value_in_topic)
        #print(count_yes)
        #print(count_no)
        
        #print(average_yes / count_yes)
        #print(average_no / count_no)
        
        
        dict_freq_values_in_topics[op] = dict_freq_value_in_topic

        op += 1
    
    print(dict_freq_values_in_topics)
    
    with open('../Save/distribution_values_in_topics_'+str(threshold_topic)+'_'+str(num_topics)+'_topics','wb') as fp:
        pickle.dump(dict_freq_values_in_topics, fp)
    
    #print("")
    
    #for i in range(len(dict_freq_values_in_topics)):
    #    for y, z in dict_freq_values_in_topics[i].items():
    #        print(z)
    #    print("")
    
    
    
    distribution_topics_in_values = {}
    for ip in range(num_topics):
        distribution_topic_in_values = {}
        for key, value in dict_freq_values_in_topics.items():
            if ip in value:
            
                distribution_topic_in_values[key] = value[ip]
       
        distribution_topics_in_values[ip] = distribution_topic_in_values
    print(distribution_topics_in_values)
        
    with open('../Save/distribution_topics_in_values_'+str(threshold_topic)+'_'+str(num_topics)+'_topics','wb') as fp:
        pickle.dump(distribution_topics_in_values, fp)
        
    print(distribution_topics_in_values)


if find_articles_co_values_per_topic == True:
    
    num_topics = 100
    threshold_topic = 0.25
    min_frequency = 0
    topics_with_marg_distr_higher_0_9 = [96, 84, 14, 42, 32, 68, 49, 25, 63, 41, 92, 67, 10, 39, 38, 79, 47, 7, 31, 99, 46, 70, 93, 97, 71, 4, 73, 18, 35, 15, 82, 66, 19, 37, 9, 72, 8, 94]
    dict_translated = {96: 1, 84: 2, 14: 3, 42: 4, 32: 5, 68: 6, 49: 7, 25: 8, 63: 9, 41: 10, 92: 11, 67: 12, 10: 13, 39: 14, 38: 15, 79: 16, 47: 17, 7: 18, 31: 19, 99: 20, 46: 21, 70: 22, 93: 23, 97: 24, 71: 25, 4: 26, 73: 27, 18: 28, 35: 29, 15: 30, 82: 31, 66: 32, 19: 33, 37: 34, 9: 35, 72: 36, 8: 37, 94: 38}
    
    
    
    value_1_searched = 2
    value_2_searched = 4
    
    topics_of_values = [49]
    number_of_topics_to_show = 1
    
       
    with open ('../Save/topics_per_value_'+str(threshold_topic)+'_'+str(num_topics)+'_topics', 'rb') as fp:
        dict_values_in_topics = pickle.load(fp)
    
    with open ('../Save/topics_of_articles_'+str(threshold_topic)+'_'+str(num_topics)+'_topics', 'rb') as fp:
        topics_of_articles = pickle.load(fp)   
        
    dict_count_freq_topics = {}
    for x in range(num_topics):
        dict_count_freq_topics[x]=0
    
    for key, value in topics_of_articles.items():
        for i in value:
            dict_count_freq_topics[i] = dict_count_freq_topics[i] + 1
  
    dict_co_values_per_topic = {}
    for x in range(num_topics):
        if x in topics_of_values:
            listy = []
            list_first_value = []
            for f in range(len(dict_values_in_topics[value_1_searched][x])):
                list_first_value.append(dict_values_in_topics[value_1_searched][x][f][0]) 
            
            list_second_value = []
            for f in range(len(dict_values_in_topics[value_2_searched][x])):
                list_second_value.append(dict_values_in_topics[value_2_searched][x][f][0]) 
            
            for h in list_first_value:
                if h in list_second_value:
                    listy.append(h)
            
            dict_co_values_per_topic[x] = listy
        

    
    
    frequency_co_values_per_topic = {}

    for key, value in dict_co_values_per_topic.items():
        #print(dict_count_freq_topics[key])
        x = len(value) / dict_count_freq_topics[key]
        frequency_co_values_per_topic[key] = x
 
    frequency_co_values_per_topic = sorted(frequency_co_values_per_topic.items(), key=operator.itemgetter(1), reverse = True)
    #print(frequency_co_values_per_topic)
       
    lista = []
    for i in frequency_co_values_per_topic:
        lista.append(i[0])
    tempy_list = lista[:number_of_topics_to_show]
    tempa_list = []
    for i in tempy_list:
        tempa_list.append(dict_translated[i])
    
    print(tempy_list)
    print(tempa_list)
    print("")
   
    for i in range(number_of_topics_to_show):
        f = dict_co_values_per_topic[frequency_co_values_per_topic[i][0]]
        
        tu = str(f).replace("', '", ") OR DOI(")
        tu = tu.replace("['", "DOI(")
        tu = tu.replace("']", ")")
        print(tu)
        
        









if create_network_data == True:
    
    num_topics = 100
    threshold_topic = 0.33
    value_researched = 0
    
    with open ('../Save/distribution_values_in_topics_'+str(threshold_topic)+'_'+str(num_topics)+'_topics', 'rb') as fp:
        dict_freq_values_in_topics = pickle.load(fp)
    
    print(dict_freq_values_in_topics)
    
    #for y, z in dict_freq_values_in_topics[value_researched].items():
    #    for j in range(len(dict_freq_values_in_topics[value_researched])):
    #        ratio = dict_freq_values_in_topics[value_researched][y] * dict_freq_values_in_topics[value_researched][j]
    #        print(ratio)
            
    x = 31        
    for y in range(31):
        x = x - 1
        for j in range(31):
            if y != j:
                ratio = dict_freq_values_in_topics[value_researched][y] * dict_freq_values_in_topics[value_researched][j]
                if ratio >= 0.1:
                    print(ratio)       
                else:
                    print("")
            




if create_Hellinger_distance_graph == True:
    
    num_topics = 100
    threshold_topic = 0.33
    
    with open ('../Save/distribution_topics_in_values_'+str(threshold_topic)+'_'+str(num_topics)+'_topics', 'rb') as fp:
        distribution_topics_in_values = pickle.load(fp)
    
    topic_0 = [[23, 92, 7, 79, 42], [78, 48, 13, 43, 18], [13, 5, 18, 25, 84], [28, 18, 14, 22, 25], [11, 16, 44, 88, 76], [27, 68, 38, 97, 96], [61, 26, 54, 34, 74], [13, 26, 18, 22, 90], [33, 54, 68, 27, 74], [41, 88, 83, 33, 68]]
    topics_200 =[[76, 46, 38, 37, 4], [78, 13, 18, 5, 58], [23, 92, 7, 79, 42], [13, 5, 18, 25, 84], [18, 14, 25, 81, 55], [68, 38, 97, 96, 70], [97, 76, 68, 85, 87], [13, 18, 5, 68, 47], [68, 9, 47, 97, 41], [41, 68, 97, 69, 37]]
    topics_200_long =[[76, 46, 38, 37, 4, 58, 81, 42, 86, 97], [78, 13, 18, 5, 58, 62, 25, 4, 72, 93], [23, 92, 7, 79, 42, 69, 93, 10, 35, 4], [13, 5, 18, 25, 84, 78, 47, 37, 75, 58], [18, 14, 25, 81, 55, 49, 53, 73, 32, 84], [68, 38, 97, 96, 70, 41, 18, 25, 46, 39], [97, 76, 68, 85, 87, 37, 95, 41, 38, 19], [13, 18, 5, 68, 47, 72, 37, 25, 84, 53], [68, 9, 47, 97, 41, 3, 87, 38, 0, 57], [41, 68, 97, 69, 37, 13, 94, 38, 19, 70]]
    topics_500 =[[46, 38, 42, 97, 70], [18, 5, 25, 93, 42], [92, 7, 79, 42, 93], [5, 18, 25, 84, 47], [18, 14, 25, 49, 73], [68, 38, 97, 96, 70], [97, 68, 41, 38, 39], [18, 5, 47, 25, 84], [68, 47, 97, 41, 38], [41, 68, 97, 38, 70]]
    
    
    
    lda = models.LdaModel.load('../../Backup/Paper analysis/5. Coherence_tests_energy/4_topics/modelLDA'+str(num_topics)+'_topics_energy(4)_onethird_coh.lda')


    distribution_topics_topics = {}
    for i in range(100):
        distribution_topic_topics = {}
        for y in range(100):
            distribution_topic_topics[y]=0
        distribution_topics_topics[i]= distribution_topic_topics
    #print(distribution_topics_topics)
    
    for f in range(100):
        for g in range(100):
            for h in topic_0:
                if f != g:
                    if all(x in h for x in [f, g]) == True:
                        #print(distribution_topics_topics[f])
                        #print(distribution_topics_topics[g][f])
                        distribution_topics_topics[f][g] = distribution_topics_topics[f][g] + 1
                        distribution_topics_topics[g][f] = distribution_topics_topics[g][f] + 1
    
    
    for f in range(100):
        for g in range(100):
            if distribution_topics_topics[f][g] >= 4:
                print(f, g)
                print(distribution_topics_topics[f][g]) 
                print('')
    print(lda.show_topics(num_topics= num_topics, num_words=10, log=True, formatted=True))             
    #for i in X:
    #    print(i)
    #X = X / X.sum(axis=1)[:, np.newaxis] # normalize vector
    #h = hellinger(X)


if get_most_cited_articles_in_topics == True:
    
    num_topics = 100
    threshold_topic = 0
    topic_researched = 99
    

    with open ('../../Backup/Paper analysis/3. Retrieval_documents_energy_sector/Output_4_topics/0.33333/scopus_txt_topics_energy(4)_DOI_onethird', 'rb') as fp:
        scopus_list_DOI = pickle.load(fp)
    
    with open ('../../Backup/Paper analysis/3. Retrieval_documents_energy_sector/Output_4_topics/0.33333/scopus_list_topics_energy(4)_citation_counts_onethird', 'rb') as fp:
        scopus_list_cit_counts = pickle.load(fp)
    
    with open ('../Save/topics_of_articles_'+str(threshold_topic)+'_'+str(num_topics)+'_topics', 'rb') as fp:
        topics_of_articles = pickle.load(fp)
        
    dict_topics_DOI_counts = {}
 
    for h in range(num_topics):
        dict_topics_DOI_counts[h]=[]
    x = 0
    for key, value in topics_of_articles.items():
        for i in value:
            try:
                integ = int(scopus_list_cit_counts[x].replace("'", ""))
            except:
                integ = 0
            listy = [scopus_list_DOI[x], integ]
            dict_topics_DOI_counts[i].append(listy)
        x += 1

    for key, value in dict_topics_DOI_counts.items():
        dict_topics_DOI_counts[key]= sorted(value, key=itemgetter(1), reverse = True)


    with open('../Save/DOIs_of_articles_per_topics_'+str(threshold_topic)+'_'+str(num_topics)+'_topics','wb') as fp:
        pickle.dump(dict_topics_DOI_counts, fp)

    lista = dict_topics_DOI_counts[topic_researched][:20]
    #print(lista)
    
 
    listy = []
    for i in lista:
        listy.append(i[0])
    #print(listy)
    
 
    ty = str(listy).replace("', '", ") OR DOI(")
    ty = ty.replace("['", "DOI(")
    ty = ty.replace("']", ")")
    print(ty)

if get_articles_most_corresponding_to_topics == True:
    
    num_topics = 100
    threshold_topic = 0.5
    length_list = 10
    topic_researched = 41
    
    with open ('../../Backup/Paper analysis/3. Retrieval_documents_energy_sector/Output_4_topics/0.33333/scopus_txt_topics_energy(4)_DOI_onethird', 'rb') as fp:
        scopus_list_DOI = pickle.load(fp)
    
  
    with open ('../Save/topics_of_articles_'+str(threshold_topic)+'_'+str(num_topics)+'_topics', 'rb') as fp:
        topics_of_articles = pickle.load(fp)
    
    lda = models.LdaModel.load('../../Backup/Paper analysis/5. Coherence_tests_energy/4_topics/modelLDA'+str(num_topics)+'_topics_energy(4)_onethird_coh.lda')
    corpus_tfidf = corpora.MmCorpus('../../Backup/Paper analysis/5. Coherence_tests_energy/4_topics/scopus_corpus'+str(num_topics)+' topics_energy(4)_onethird_coh.mm')
    dictionary = corpora.Dictionary.load('../../Backup/Paper analysis/5. Coherence_tests_energy/4_topics/scopus_list'+str(num_topics)+'_energy(4)_onethird_coh.dict')
    
    all_topics = lda.get_document_topics(corpus_tfidf, minimum_probability=0, per_word_topics = False)

    with open ('../../Backup/Paper analysis/3. Retrieval_documents_energy_sector/Output_4_topics/0.33333/scopus_list_topics_energy(4)_citation_counts_onethird', 'rb') as fp:
        scopus_list_cit_counts = pickle.load(fp)
    
    dict_topics_DOI_strengh = {}
    x = 0
    for key, value in topics_of_articles.items():
        for j in value:        
            if j == topic_researched:
                dict_topics_DOI_strengh[scopus_list_DOI[x]]= all_topics[x][j][1] 
        x += 1
    
    dict_topics_DOI_strengh = sorted(dict_topics_DOI_strengh.items(), key=operator.itemgetter(1), reverse = True)
    print(dict_topics_DOI_strengh [:length_list])

    lista = dict_topics_DOI_strengh[:length_list]
    
    listy = []
    for i in lista:
        listy.append(i[0])
 
    ty = str(listy).replace("', '", ") OR DOI(")
    ty = ty.replace("['", "DOI(")
    ty = ty.replace("']", ")")
    print(ty)

    
if get_most_cited_articles_in_topics_per_value == True:
    
    num_topics = 100
    value_searched = 6
    threshold_topic = 0
    limit = 5000
    #topics_to_search_in = [97, 68, 37, 41, 38, 19, 39, 9, 63, 47]
    #topics_to_search_in = [97, 68, 67, 7, 37, 38, 47, 35, 39, 70]
    topics_to_search_in = [0,59]
    #topics_to_search_in = range(0,100,1)
    
    with open ('../Save/DOIs_of_articles_per_topics_'+str(threshold_topic)+'_'+str(num_topics)+'_topics', 'rb') as fp:
        dict_topics_DOI_counts = pickle.load(fp)
    
    with open ('../../Backup/Paper analysis/3. Retrieval_documents_energy_sector/Output_4_topics/0.33333/scopus_txt_topics_energy(4)_onethird', 'rb') as fp:
        scopus_list_txt = pickle.load(fp)
    
    with open ('../../Backup/Paper analysis/3. Retrieval_documents_energy_sector/Output_4_topics/0.33333/scopus_txt_topics_energy(4)_DOI_onethird', 'rb') as fp:
        scopus_list_DOI = pickle.load(fp)
    
    list_of_articles_in_topics_discussing_value = []
    for h in topics_to_search_in:
    
        v = 0
        list_of_articles_in_topic_discussing_value = []
        listu = dict_topics_DOI_counts[h]
        for f in listu:
            if not not f:
                x = scopus_list_DOI.index(f[0])
                y = scopus_list_txt[x]
                        
                if len([i for i in values[value_searched] if i in y]) > 0:
                    list_of_articles_in_topic_discussing_value.append(f)
        
        list_of_articles_in_topics_discussing_value.append(list_of_articles_in_topic_discussing_value)
    
    
    for h in list_of_articles_in_topics_discussing_value:
    
        listrg = []
        for g in h:
            listrg.append(g[0])
                 
        tu = str(listrg[:limit]).replace("', '", ") OR DOI(")
        tu = tu.replace("['", "DOI(")
        tu = tu.replace("']", ")")
        print(tu)
    
    
    
    
    
    
    
    
    
    
if get_all_values_topics == True:

    num_topics = 100
    
    value_1 = 6
    value_2 = 6
    
    make_data = False
    get_data = True
    
    if make_data == True:
        threshold_topic = [0.33, 0.5]
        for gt in threshold_topic:
        #with open ('../Save/'+str(version)+'_'+str(threshold_calc)+'/DOI_list_value_threshold_'+str(threshold_calc)+'_'+str(value_1)+'_and_'+str(value_2), 'rb') as fp:
        #    DOI_list = pickle.load(fp)
            
            with open ('../../Backup/Paper analysis/3. Retrieval_documents_energy_sector/Output_4_topics/0.33333/scopus_txt_topics_energy(4)_DOI_onethird', 'rb') as fp:
                scopus_list_DOI = pickle.load(fp)
                
            #with open ('../../Backup/Paper analysis/3. Retrieval_documents_energy_sector/Output_4_topics/0.33333/scopus_txt_topics_energy(4)_onethird', 'rb') as fp:
            #    scopus_list_txt = pickle.load(fp)
            
            lda = models.LdaModel.load('../../Backup/Paper analysis/5. Coherence_tests_energy/4_topics/modelLDA'+str(num_topics)+'_topics_energy(4)_onethird_coh.lda')
            corpus_tfidf = corpora.MmCorpus('../../Backup/Paper analysis/5. Coherence_tests_energy/4_topics/scopus_corpus'+str(num_topics)+' topics_energy(4)_onethird_coh.mm')
            dictionary = corpora.Dictionary.load('../../Backup/Paper analysis/5. Coherence_tests_energy/4_topics/scopus_list'+str(num_topics)+'_energy(4)_onethird_coh.dict')
        
            all_topics = lda.get_document_topics(corpus_tfidf, minimum_probability=0, per_word_topics = False)
            #print(lda.show_topics(num_topics= num_topics, num_words=10, log=True, formatted=True))
                
            dict_articles_per_topic = {}    
            for i in range(num_topics):
                dict_articles_per_topic[i]=[]
            
            #print(dict_articles_per_topic)
            
            f = 0
            for i in all_topics:
                for a in i:
                    if a[1] > gt:
                        dict_articles_per_topic[a[0]].append(scopus_list_DOI[f])
                        
                f += 1
                #print(dict_articles_per_topic)
                if f % 1000 == 0:
                    
                    print(f)
            
            with open('../Save/dict_articles_per_topic'+str(gt),'wb') as fp:
                pickle.dump(dict_articles_per_topic, fp)

    if get_data == True:
         
         
        threshold_topic = 0.5
        topic = 46
        
        with open ('../Save/dict_articles_per_topic'+str(threshold_topic), 'rb') as fp:
            dict_articles_per_topic = pickle.load(fp)
    
        print(len(dict_articles_per_topic[topic]))
        print(dict_articles_per_topic[topic])

if get_topics_of_articles == True:
    print('Working on get_topics_of_articles...')
    num_topics = 100
    
    value_1 = 1
    value_2 = 5
    
    with open ('../Save/'+str(version)+'_'+str(threshold_calc)+'/DOI_list_value_threshold_'+str(threshold_calc)+'_'+str(value_1)+'_and_'+str(value_2), 'rb') as fp:
        DOI_list = pickle.load(fp)
        #print(DOI_list)

    with open ('../Save/'+str(version)+'_'+str(threshold_calc)+'/scopus_list_txt_energy(4)_threshold_'+str(threshold_calc)+'_'+str(value_1)+'_'+str(value_2), 'rb') as fp:
        dict_speci_co_values_reduced_threshold = pickle.load(fp)

    with open ('../Save/'+str(version)+'_'+str(threshold_calc)+'/scopus_list_txt_energy(4)_top_100_'+str(value_1)+'_'+str(value_2), 'rb') as fp:
        dict_speci_co_values_reduced_top_100 = pickle.load(fp)


    lda = models.LdaModel.load('../../Backup/Paper analysis/5. Coherence_tests_energy/4_topics/modelLDA'+str(num_topics)+'_topics_energy(4)_onethird_coh.lda')
    corpus_tfidf = corpora.MmCorpus('../../Backup/Paper analysis/5. Coherence_tests_energy/4_topics/scopus_corpus'+str(num_topics)+' topics_energy(4)_onethird_coh.mm')
    dictionary = corpora.Dictionary.load('../../Backup/Paper analysis/5. Coherence_tests_energy/4_topics/scopus_list'+str(num_topics)+'_energy(4)_onethird_coh.dict')

    all_topics = lda.get_document_topics(corpus_tfidf, minimum_probability=0, per_word_topics = False)
    #print(lda.show_topics(num_topics= num_topics, num_words=10, log=True, formatted=True))

    
    dict_co_value_to_topic = {}
    for i in range(num_topics):
        dict_co_value_to_topic[i]=0
    
    
    for i in range(len(dict_speci_co_values_reduced_threshold)):
        for h in all_topics[dict_speci_co_values_reduced_threshold[i][0]]:
            dict_co_value_to_topic[h[0]] = dict_co_value_to_topic[h[0]] + h[1]
            
            
    print(len(dict_speci_co_values_reduced_threshold))               
    
    name_topics = []
    for i in range(num_topics):
        name_topics.append('Topic '+str(i))

    y_pos = np.arange(len(name_topics))
    scores = []
    for i in range(len(dict_co_value_to_topic)):
        topic_score = dict_co_value_to_topic[i]
        scores.append(topic_score)
    
    plt.bar(y_pos, scores, align='center', alpha=0.5)
    plt.xticks(y_pos, name_topics, rotation = 90)
    plt.ylabel('Frequency')
    plt.xlabel('Topics')
    plt.title('Frequency of topics in co-value '+str(value_1)+' and '+str(value_2))
    plt.show()











    
if values_per_topic == True:
    
    print('Working on values_per_topic...')
    num_topics = 100
    
    
    dict_values_to_topics = {}
    for i in range(num_topics):
        dict_values_to_topic = {}
        for h in range(len(values_names)):
            dict_values_to_topic[h] = 0
            
        dict_values_to_topics[i] = dict_values_to_topic
            
    
        
    
    with open ('../Save/'+str(version)+'_'+str(threshold_calc)+'/scopus_list_txt_energy(4)_articles_to_values', 'rb') as fp:
        dict_articles_values = pickle.load(fp)    
        
    lda = models.LdaModel.load('../../Backup/Paper analysis/5. Coherence_tests_energy/4_topics/modelLDA'+str(num_topics)+'_topics_energy(4)_onethird_coh.lda')
    corpus_tfidf = corpora.MmCorpus('../../Backup/Paper analysis/5. Coherence_tests_energy/4_topics/scopus_corpus'+str(num_topics)+' topics_energy(4)_onethird_coh.mm')
    dictionary = corpora.Dictionary.load('../../Backup/Paper analysis/5. Coherence_tests_energy/4_topics/scopus_list'+str(num_topics)+'_energy(4)_onethird_coh.dict')
    
    all_topics = lda.get_document_topics(corpus_tfidf, minimum_probability=0, per_word_topics = False)
        
    new_dictionary = copy.deepcopy(all_topics)
        #print(all_topics[0]) 
        
        #dict_values_to_topics = {}
        #itr1 = 0
        
        
        
    for f in range(len(dict_articles_values)):
            #print(dict_articles_values[f+1])
        for x in range(len(values_names)):
            
            h = sum(dict_articles_values[f+1][x].values())
            dict_articles_values[f+1][x] = h
            
    
        
        
    for f in range(len(dict_articles_values)):
        for x in range(len(values_names)):
            if dict_articles_values[f+1][x] >= threshold_calc:
                #print(dict_articles_values[f+1])
                #print(dict_values_to_topics)
                for h in range(num_topics):
                    val = new_dictionary[f][h][1]
                    #print(val)
                    dict_values_to_topics[h][x] = dict_values_to_topics[h][x] + val
                #print(dict_values_to_topics)
                #print("")
        if f % 1 == 0:
            print(f)
            print(new_dictionary[f])
            print(dict_values_to_topics)
            print("")
    #print(dict_values_to_topics)
            
    with open ('../Save/'+str(version)+'_'+str(threshold_calc)+'/dict_values_to_topics', 'rb') as fp:
        dict_values_to_topics = pickle.load(fp)
            
    

            
    
    
    
    #dict_values_to_topics = {}
    #itr1 = 0
    #for h in range(num_topics):
    #    itr2 = 0
    #    dict_values_to_topic = {}
    #    for x in range(len(values_names)):
    #        temp = 0    
    #        for f in range(len(dict_articles_values)):
    #            val = 0
    #            
    #            if sum(dict_articles_values[f+1][x].values()) >= threshold_calc:
    #                val = all_topics[f][h][1]
                    #print(val)
    #            temp = temp + val
                #print(temp)
    #        dict_values_to_topic[itr2] = temp
    #        itr2 += 1
     #       print(dict_values_to_topic)
    #    dict_values_to_topics[itr1] = dict_values_to_topic           
    #    itr1 += 1                    
    #    print(dict_values_to_topics)        
               

                
                
        
       
    
    
    #print(dict_articles_values[1])
    #dict_value_to_topic = {}
    #for i in range(num_topics):
    #    dict_value_to_topic[i]=0
    
    
    
    #for i in range(len(dict_speci_co_values_reduced_threshold)):
    #    for h in all_topics[dict_speci_co_values_reduced_threshold[i][0]]:
    #        dict_co_value_to_topic[h[0]] = dict_co_value_to_topic[h[0]] + h[1]
    
    