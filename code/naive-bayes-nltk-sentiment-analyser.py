import os
import json
from csv import DictReader, DictWriter
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from sklearn.model_selection import train_test_split
from nltk.stem.porter import *
from nltk import word_tokenize
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

SEED = 5



if __name__ == "__main__":

    # Read in data

    dataset_x = []
    dataset_y = []
    pos_data =[]
    neg_data =[]
    test_data = []
    test_ids = [] 
    pos_data_Count=0
    neg_data_Count=0
    i=0
    with open('hotelNegT-train.txt',encoding="utf8") as f:
        for data in f.readlines():    
            s = (data.split('\t',1)[1].lower())
            s=re.sub('\d', '', s)
            s=re.sub(r'\(', '', s)
            s=re.sub(r'\)', '', s)
            dataset_x.append(s)
            dataset_y.append(-1)
            i+=1
    with open('hotelPosT-train.txt',encoding="utf8") as f:
        for data in f.readlines():    
            s = (data.split('\t',1)[1].lower())
            s=re.sub('\d', '', s)
            s=re.sub(r'\(', '', s)
            s=re.sub(r'\)', '', s)
            dataset_x.append(s)
            dataset_y.append(1)
            i+=1
    with open('test.txt',encoding="utf8") as f:
        for data in f.readlines(): 
            s = (data.split('\t',1)[1].lower())
            s=re.sub('\d', '', s)
            s=re.sub(r'\(', '', s)
            s=re.sub(r'\)', '', s)
            test_data.append(s)
            test_ids.append(data.split('\t',1)[0])
            
    X_test = test_data
    
    X_train, y_train = dataset_x, dataset_y

    stemmer = PorterStemmer()
    
    print("Training Classifier")
    vocab = {}
    positive_word_dictionary= {}
    negative_word_dictionary= {}
    
    for i in range(len(X_train)):
        
        words = re.split(r'\s+|\.|\!|,', X_train[i])
        words = [i for i in words if i not in stop]
        words = [i for i in words if len(i.strip('')) !=0]
        words = [stemmer.stem(word) for word in words]
        
        pos = y_train[i] == -1 
        if pos == True:
            neg_data.append(X_train[i])
            neg_data_Count+= len(words)
        else:
            pos_data.append(X_train[i])  
            pos_data_Count+= len(words)

        for string in words:
            vocab[string] = 1
            if pos == True:
                if string in negative_word_dictionary.keys():
                    negative_word_dictionary[string] +=1
                else:
                    negative_word_dictionary[string] = 1
            else:
                if string in positive_word_dictionary.keys():
                    positive_word_dictionary[string] +=1
                else:
                    positive_word_dictionary[string] = 1
    V = len(vocab)
    print("vocab length = ",V)
    print("neg_data_Count",neg_data_Count)
    print("pos_data_Count",pos_data_Count)
    Nd= len(X_train)
    Ncpos = len(pos_data)
    Ncneg = len(neg_data)
    logpriorpos = math.log(Ncpos/Nd)
    logpriorneg = math.log(Ncneg/Nd)
    log_likelihoodpos= {}
    log_likelihoodneg = {}
    print(positive_word_dictionary,negative_word_dictionary)
    for w in vocab.keys():
        pw,nw =1,1
        if w in positive_word_dictionary.keys(): 
            pw += positive_word_dictionary[w]
        log_likelihoodpos[w] = math.log(pw/(pos_data_Count + V))
        if w in negative_word_dictionary.keys():
            nw += negative_word_dictionary[w]
        nw = nw/(neg_data_Count + V)
        log_likelihoodneg[w] = math.log(nw)

    print("Training Classifier")
    y_pred = []
    output = open("output2.txt",'w')
    print(logpriorpos,logpriorneg)
    for j in range(len(X_test)):
        test = X_test[j]
        max_pos = logpriorpos
        max_neg = logpriorneg
        words = re.split(r'\s+|\.|\!|,',test)
        words = [i for i in words if i not in stop]
        words = [i for i in words if len(i.strip(' ') )!=0]
        
        stemmed_words =[]
        for word in words:
            stemmed_words.append(stemmer.stem(word))
        stemmed_words = set(stemmed_words)
        for w in stemmed_words:
            if w in vocab:
                print(w,log_likelihoodpos[w],log_likelihoodneg[w])
                max_pos += log_likelihoodpos[w]
                max_neg += log_likelihoodneg[w]
        print(max_pos,max_neg)
        if max_pos > max_neg:
            y_pred.append(1)
        else:
            y_pred.append(-1)
        if y_pred[j] == 1:
            print(test_ids[j],"POS")
            output.write(test_ids[j]+"\t"+"POS\n")
        else :
            print(test_ids[j],"NEG")
            output.write(test_ids[j]+"\t"+"NEG\n")