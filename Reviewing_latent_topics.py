'''
Created on 8 dec. 2016

@author: tewdewildt
'''
#import csvkit
import csv
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
#from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import re
from six import iteritems
import pickle
import logging
import gensim
from gensim import corpora, models, similarities
from pprint import pprint


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#csv.field_size_limit(500 * 1024 * 1024)

''' 
Import data from the csv file and put into a dictionary 

CSV file should include columns named 'Abstract', 'Author Keywords', 'Title' and 'DOI'.
One row should be one article
'''

with open('./data/scopus.csv', encoding="utf8", errors='ignore') as infile:
    reader = csv.DictReader(infile)
    data = {}
    for row in reader:
        for header, value in row.items():
            try:
                data[header].append(value)
            except KeyError:
                data[header] = [value]


''' 
Group abstract, keywords and title into one tokenized string 
'''

tokenizer = RegexpTokenizer(r'\w+')
nltk.download('stopwords')
stopwords = stopwords.words('english')
p_stemmer = PorterStemmer()

text_of_articles = []
i = 0

for row in data['Abstract']:
    text_of_article = str(data['Abstract'][i]+' '+data['Author Keywords'][i]+' '+data['Title'][i])
    text_of_article = ''.join(text_of_article)
    text_of_article = re.sub(r'"' or r'[', ' ', text_of_article)
    text_of_article = re.sub(r'http\S+', '', text_of_article)
    text_of_article = re.sub('\W+',' ', text_of_article)
    text_of_article = text_of_article.strip("\t").strip("abstract available]")
    text_of_article = text_of_article.lower()
    text_of_article = ''.join([a for a in text_of_article if not a.isdigit()])
    
    tokens = tokenizer.tokenize(text_of_article)
    stopped_tokens = [h for h in tokens if not h in stopwords]
    for f in stopped_tokens:
        if(len(f))<=1:
            stopped_tokens.remove(f)
    
    # stem tokens
    #stemmed_tokens = [p_stemmer.stem(h) for h in stopped_tokens]
    
    text_of_articles.append(stopped_tokens)
    i += 1
  
''' 
Make a topic model 
'''
    
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s : %(levelname)s : %(message)s', 
                    datefmt='%m-%d %H:%M',
                    filename='../save/log_topic_model_creation.log',
                    filemode='w')

number_topics_to_be_found = 5
number_passes = 5
num_words_to_show = 10

dictionary = corpora.Dictionary(text_of_articles)
dictionary.compactify()  # remove gaps in id sequence after words that were removed
dictionary.save('../save/gensim_dictionary.dict')
            
corpus = [dictionary.doc2bow(text) for text in text_of_articles]
corpora.MmCorpus.serialize('../save/gensim_corpus.mm', corpus)
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]     
                       
lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics = number_topics_to_be_found, passes = number_passes)
lda.save('../save/gensim_LDA_model.lda')

''' 
Create dictionary with DOI as key and text as value for articles only related to selected topics
'''

topics_of_interest = [0, 1]
threshold = 0.25
doc_lda = lda[corpus_tfidf]
dict_articles_selected_topics = {}

i = 0
for o in doc_lda:
    counter = 0
    for x in o:
        if x[0] in topics_of_interest and x[1] >= threshold:
            counter += 1
    if counter > 0:
        dict_articles_selected_topics[data['DOI'][i]]= text_of_articles[i]
    i += 1

''' 
Select articles addressing value for selected topics
'''

dict_articles_selected_topics_selected_values = {}

justice = ['equity', 'inequity', 'fair', 'unfair', 'justice', 'injustice', 'impartial', 'unbiased', 'objectivity', 'lawful', 'unlawful', 'egalitarian', 'inegalitarian', 'distributive', 'fairness', 'justness', 'impartiality', 'equitable']

for key, value in dict_articles_selected_topics.items():
    if any(x in value for x in justice):
        dict_articles_selected_topics_selected_values[key] = value
    

for key, value in dict_articles_selected_topics_selected_values.items():
    print(key)



