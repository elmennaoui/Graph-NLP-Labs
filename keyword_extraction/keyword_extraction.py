import os
import string
import re 
import operator
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from library import clean_text_simple,terms_to_graph,core_dec,accuracy_metrics

stemmer = nltk.stem.PorterStemmer()
stpwds = stopwords.words('english')
punct = string.punctuation.replace('-', '')

##################################
# read and pre-process abstracts #
##################################

path_to_abstracts = "/data/Hulth2003testing/abstracts/"
abstract_names = sorted(os.listdir(path_to_abstracts))

abstracts = []
for counter,filename in enumerate(abstract_names):
    # read file
    with open(path_to_abstracts + '/' + filename, 'r') as my_file: 
        text = my_file.read().splitlines()
    text = ' '.join(text)
    # remove formatting
    text = re.sub('\s+', ' ', text)
    abstracts.append(text)
    
    if counter % round(len(abstract_names)/5) == 0:
        print(counter, 'files processed')

abstracts_cleaned = []
for counter,abstract in enumerate(abstracts):
    my_tokens = clean_text_simple(abstract,my_stopwords=stpwds,punct=punct)
    abstracts_cleaned.append(my_tokens)
    
    if counter % round(len(abstracts)/5) == 0:
        print(counter, 'abstracts processed')

###############################################
# read and pre-process gold standard keywords #
###############################################

path_to_keywords = "/home/med/Bureau/ALTEGRAD/lab1/data/Hulth2003testing/uncontr/"### fill me! ###
keywd_names = sorted(os.listdir(path_to_keywords))
   
keywds_gold_standard = []

for counter,filename in enumerate(keywd_names):
    # read file
    with open(path_to_keywords + filename, 'r') as my_file: 
        text = my_file.read().splitlines()
    text = ' '.join(text)
    text =  re.sub('\s+', ' ', text) # remove formatting
    text = text.lower()
    # turn string into list of keywords, preserving intra-word dashes 
    # but breaking n-grams into unigrams
    keywds = text.split(';')
    keywds = [keywd.strip().split(' ') for keywd in keywds]
    keywds = [keywd for sublist in keywds for keywd in sublist] # flatten list
    keywds = [keywd for keywd in keywds if keywd not in stpwds] # remove stopwords (rare but may happen due to n-gram breaking)
    keywds_stemmed = [stemmer.stem(keywd) for keywd in keywds]
    keywds_stemmed_unique = list(set(keywds_stemmed)) # remove duplicates (may happen due to n-gram breaking)
    keywds_gold_standard.append(keywds_stemmed_unique)
    
    if counter % round(len(keywd_names)/5) == 0:
        print(counter, 'files processed')

##############################
# precompute graphs-of-words #
##############################

gs = []
for abstract in abstracts_cleaned:
    gs.append(terms_to_graph(abstract, 4))

##################################
# graph-based keyword extraction #
##################################

my_percentage = 0.33 # for PR and TF-IDF

method_names = ['kc','wkc','pr','tfidf']
keywords = dict(zip(method_names,[[],[],[],[]]))

for counter,g in enumerate(gs):
    # k-core
    kcore = core_dec(g,False)
    core_numbers = list(kcore.items())

    keywords['kc'].append([tuple[0] for tuple in core_numbers if tuple[1] == max(kcore.values())])
    # weighted k-core
    ### fill the gaps (repeat the procedure used for k-core) ###
    wkcore = core_dec(g,True)
    weighted_core_numbers = list(wkcore.items())

    keywords['wkc'].append([tuple[0] for tuple in weighted_core_numbers if tuple[1] == max(wkcore.values())])

    # PageRank
    pr_scores = zip(g.vs['name'],g.pagerank())
    pr_scores = sorted(pr_scores, key=operator.itemgetter(1), reverse=True) # in decreasing order
    numb_to_retain = int(len(pr_scores)*my_percentage) # retain top 'my_percentage' % words as keywords
    keywords['pr'].append([tuple[0] for tuple in pr_scores[:numb_to_retain]])
        
    if counter % round(len(gs)/5) == 0:
        print(counter)

#############################
# TF-IDF keyword extraction #
#############################

abstracts_cleaned_strings = [' '.join(elt) for elt in abstracts_cleaned] # to ensure same pre-processing as the other methods
tfidf_vectorizer = TfidfVectorizer(stop_words=stpwds)
doc_term_matrix = tfidf_vectorizer.fit_transform(abstracts_cleaned_strings)
terms = tfidf_vectorizer.get_feature_names()
vectors_list = doc_term_matrix.todense().tolist()

for counter,vector in enumerate(vectors_list):
    terms_weights = zip(terms,vector) # bow feature vector as list of tuples
    nonzero = [t for t in terms_weights if t[1] != 0]
    nonzero = sorted(nonzero, key=operator.itemgetter(1), reverse=True) # in decreasing order
    numb_to_retain = int(len(nonzero)*my_percentage) # retain top 'my_percentage' % words as keywords
    keywords['tfidf'].append([tuple[0] for tuple in nonzero[:numb_to_retain]])
    
    if counter % round(len(vectors_list)/5) == 0:
        print(counter)

##########################
# performance comparison #
##########################

perf = dict(zip(method_names,[[],[],[],[]]))

for idx,truth in enumerate(keywds_gold_standard):
    for mn in method_names:
        perf[mn].append(accuracy_metrics(keywords[mn][idx], truth))

lkgs = len(keywds_gold_standard)

# print macro-averaged results (averaged at the collection level)
for k,v in perf.items():
    print(k + ' performance: \n')
    print('precision:', round(100*sum([tuple[0] for tuple in v])/lkgs,2))
    print('recall:', round(100*sum([tuple[1] for tuple in v])/lkgs,2))
    print('F-1 score:', round(100*sum([tuple[2] for tuple in v])/lkgs,2))
    print('\n')
