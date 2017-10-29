import os
import json
from csv import DictReader, DictWriter
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from sklearn.model_selection import train_test_split

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
            
            dataset_x.append(data.split('\t', 1)[1])
            dataset_y.append(-1)
            i+=1
    with open('hotelPosT-train.txt',encoding="utf8") as f:
        for data in f.readlines():    
            dataset_x.append(data.split('\t', 1)[1])
            dataset_y.append(1)
            i+=1
    with open('test.txt',encoding="utf8") as f:
        for data in f.readlines(): 
            test_data.append(data.split('\t',1)[1])
            test_ids.append(data.split('\t',1)[0])
            i+=1
    X_test = test_data
    print("total dataset size = ",i)
    print(len(dataset_x),len(dataset_y))
    X_train, y_train = dataset_x, dataset_y

    print("Training Classifier")
    vocab = {}
    positive_word_dictionary= {}
    negative_word_dictionary= {}
    
    for i in range(len(X_train)):
        words = re.split('\s+', X_train[i])
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
    output = open("output1.txt",'w')
    for j in range(len(X_test)):
        test = X_test[j]
        max_pos = logpriorpos
        max_neg = logpriorneg
        words = re.split('\s+',test)
        for w in words:
            if w in vocab:
                max_pos += log_likelihoodpos[w]
                max_neg += log_likelihoodneg[w]
        print(max_neg,max_pos)
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