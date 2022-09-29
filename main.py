from pipeline import eval 
from pipeline import LogReg, NaiveBayes, RndFrst, SVM, DecTree
from pipeline2 import LogReg2, NaiveBayes2, RndFrst2, SVM2, DecTree2
from data_clean import process_df, clean, tknzr, stmr, lmtzr 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import nltk
from nltk.tokenize import RegexpTokenizer, TreebankWordTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import ngrams

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

from sklearn.model_selection import GridSearchCV, train_test_split

from collections import Counter

from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

data = pd.read_csv('combined_data.csv')

labels = {'negative':0, 'neutral':1, 'positive':2}
data['label'] = [labels[item] for item in data.sntmt]

choice = int (input("Enter your choice of feature extraction technique:"
                   "\n 1 - CountVectorizer"
                   "\n 2 - TfidfVectorizer"
                   "\n 3 - Word2Vec \n"))

#no change to dataset
data_og = process_df(data.copy(), tknzr, rem_sw='True')
#dataset after stemming 
data_st = process_df(data.copy(), stmr, rem_sw='True')
#dataset after lemmatization
data_lm = process_df(data.copy(), lmtzr, rem_sw='True')

DATASETS = [data_og, data_st, data_lm]

n_grm = [(1,1),(2,2),(3,3),(1,2),(2,3)]
clfrs = ['LogReg','RndFrst','NaiveBayes','SVM','DecTree']
clfrs2 = ['LogReg2','RndFrst2','SVM2','DecTree2']
features = [2000, 3000]

RESULTS_COUNTVEC = RESULTS_TFIDF = {'Pre-Processing Technique':[],
                                    'N_Grams':[],
                                    'No_of_Features':[],
                                    'Classifier Applied':[],
                                    'Accuracy':[],
                                    'Precision':[],
                                    'Recall':[],
                                    'F-Score':[]
                                    }

RESULTS_W2V = RESULTS_D2V = {'Pre-Processing Technique':[],
                            'Model':[],
                            'No_of_Features':[],
                            'Classifier Applied':[],
                            'Accuracy':[],
                            'Precision':[],
                            'Recall':[],
                            'F-Score':[]
                            }

if choice == 1:
    
    for ds in DATASETS:

        for ng in n_grm:

            count_vec = CountVectorizer(ngram_range=ng)
            count_data = count_vec.fit_transform(ds.processed_text)
            count_vec_df = pd.DataFrame(count_data.toarray(), columns = count_vec.get_feature_names_out())

            x = count_vec_df.values
            y = ds['label'].values

            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=15)

            for clfr in clfrs:

                if clfr == 'LogReg':
                    clf = LogReg
                elif clfr == 'RndFrst':
                    clf = RndFrst
                elif clfr == 'NaiveBayes':
                    clf = NaiveBayes
                elif clfr == 'SVM':
                    clf = SVM
                elif clfr == 'DecTree':
                    clf = DecTree
                    
                for nf in features:

                    print("########## RESULTS FOR {} MODEL, {} NGRAMS, {} FEATURES  ##########".format(clfr, ng, nf))
                    y_pred = clf(X_train, X_test, y_train, y_test, nf)
                    a,p,r,f = eval(y_test, y_pred)

                    if ds.equals(data_og):
                        RESULTS_COUNTVEC['Pre-Processing Technique'].append('None')
                    elif ds.equals(data_st):
                        RESULTS_COUNTVEC['Pre-Processing Technique'].append('Stemming')
                    elif ds.equals(data_lm):
                        RESULTS_COUNTVEC['Pre-Processing Technique'].append('Lemmatizing')
                    RESULTS_COUNTVEC['N_Grams'].append(ng)
                    RESULTS_COUNTVEC['No_of_Features'].append(nf)
                    RESULTS_COUNTVEC['Classifier Applied'].append(clfr)
                    RESULTS_COUNTVEC['Accuracy'].append(a)
                    RESULTS_COUNTVEC['Precision'].append(p)
                    RESULTS_COUNTVEC['Recall'].append(r)
                    RESULTS_COUNTVEC['F-Score'].append(f)

    results_cntvec = pd.DataFrame(RESULTS_COUNTVEC, index=None)
    sorted_results_cntvec = results_cntvec.sort_values(by=['Accuracy'], ascending = False)
    sorted_results_cntvec.to_csv('./Results/CountVec_Results.csv')


elif choice == 2:
    
    for ds in DATASETS:

        for ng in n_grm:

            tfidf_vec = TfidfVectorizer(ngram_range=ng)
            tfidf_data = tfidf_vec.fit_transform(ds.processed_text)
            tfidf_vec_df = pd.DataFrame(tfidf_data.toarray(), columns = tfidf_vec.get_feature_names_out())

            x = tfidf_vec_df.values
            y = ds['label'].values

            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=15)

            for clfr in clfrs:

                if clfr == 'LogReg':
                    clf = LogReg
                elif clfr == 'RndFrst':
                    clf = RndFrst
                elif clfr == 'NaiveBayes':
                    clf = NaiveBayes
                elif clfr == 'SVM':
                    clf = SVM
                elif clfr == 'DecTree':
                    clf = DecTree
                    
                for nf in features:

                    print("########## RESULTS FOR {} MODEL, {} NGRAMS, {} FEATURES  ##########".format(clfr, ng, nf))
                    y_pred = clf(X_train, X_test, y_train, y_test, nf)
                    a,p,r,f = eval(y_test, y_pred)

                    if ds.equals(data_og):
                        RESULTS_TFIDF['Pre-Processing Technique'].append('None')
                    elif ds.equals(data_st):
                        RESULTS_TFIDF['Pre-Processing Technique'].append('Stemming')
                    elif ds.equals(data_lm):
                        RESULTS_TFIDF['Pre-Processing Technique'].append('Lemmatizing')
                    RESULTS_TFIDF['N_Grams'].append(ng)
                    RESULTS_TFIDF['No_of_Features'].append(nf)
                    RESULTS_TFIDF['Classifier Applied'].append(clfr)
                    RESULTS_TFIDF['Accuracy'].append(a)
                    RESULTS_TFIDF['Precision'].append(p)
                    RESULTS_TFIDF['Recall'].append(r)
                    RESULTS_TFIDF['F-Score'].append(f)

    results_tfidf = pd.DataFrame(RESULTS_TFIDF, index=None)
    sorted_results_tfidf = results_tfidf.sort_values(by=['Accuracy'], ascending = False)
    sorted_results_tfidf.to_csv('./Results/Tfidf_Results.csv')

elif choice == 3:
    features2 = [400, 800]
    def word_vector(tokens, size):
        vec = np.zeros(size).reshape((1, size))
        count = 0
        for word in tokens:
            try:
                vec += model_w2v.wv[word].reshape((1, size))
                count += 1.
            except KeyError:  # handling the case where the token is not in vocabulary
                continue
        if count != 0:
            vec /= count
        return vec

    for ds in DATASETS:

        for sg in [0,1]:
        
            ds['tokens'] = ds['processed_text'].map(lambda txt: tknzr(txt, False).split(" "))

            tokens = pd.Series(ds['tokens']).values

            model_w2v = Word2Vec(
                tokens,
                vector_size = 1000, # desired no. of features/independent variables
                window = 3, # context window size
                min_count = 1, # Ignores all words with total frequency lower than 2.                                  
                sg = sg,
                hs = 0,
                negative = 10, # for negative sampling
                workers = 32, # no.of cores
                seed = 34
            )

            wordvec_arrays = np.zeros((len(tokens), 1000)) 
            for i in range(len(tokens)):
                wordvec_arrays[i,:] = word_vector(tokens[i], 1000)
            wordvec_df = pd.DataFrame(wordvec_arrays)

            x = wordvec_df.values
            y = ds['label'].values

            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=15)

            for clfr in clfrs2:

                if clfr == 'LogReg':
                    clf = LogReg2
                elif clfr == 'RndFrst':
                    clf = RndFrst2
                elif clfr == 'SVM':
                    clf = SVM2
                elif clfr == 'DecTree':
                    clf = DecTree2

                for nf in features2:
                    print("########## RESULTS FOR {} CLASSIFIER, {} FEATURES  ##########".format(clfr, nf))
                    y_pred = clf(X_train, X_test, y_train, y_test, nf)
                    a,p,r,f = eval(y_test, y_pred)

                    if ds.equals(data_og):
                        RESULTS_W2V['Pre-Processing Technique'].append('None')
                    elif ds.equals(data_st):
                        RESULTS_W2V['Pre-Processing Technique'].append('Stemming')
                    elif ds.equals(data_lm):
                        RESULTS_W2V['Pre-Processing Technique'].append('Lemmatizing')
                    
                    if sg == 0:
                        RESULTS_W2V['Model'].append('CBOW')
                    else:
                        RESULTS_W2V['Model'].append('Skipgram')

                    RESULTS_W2V['No_of_Features'].append(nf)
                    RESULTS_W2V['Classifier Applied'].append(clfr)
                    RESULTS_W2V['Accuracy'].append(a)
                    RESULTS_W2V['Precision'].append(p)
                    RESULTS_W2V['Recall'].append(r)
                    RESULTS_W2V['F-Score'].append(f)
                    
    results_w2v = pd.DataFrame(RESULTS_W2V, index=None)
    sorted_results_w2v = results_w2v.sort_values(by=['Accuracy'], ascending = False)
    sorted_results_w2v.to_csv('./Results/W2V_Results.csv')
    
else:
    print("Please enter a valid option!")        