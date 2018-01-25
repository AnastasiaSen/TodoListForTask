# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 21:15:39 2018

@author: Ася
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import csv

news_train = list(csv.reader(open('news_train.txt', 'rt', encoding="utf8"), delimiter='\t'))

count = 1
news_train_data = []
for x in news_train:
    count+=1
    news_train_data.append(x[2])
count = 1
news_train_data_target = []
for x in news_train:
    count+=1
    news_train_data_target.append(x[0])
    

news_final = list(csv.reader(open('news_test.txt', 'rt', encoding="utf8"), delimiter='\t'))

news_data_final = []
count = 1
for x in news_final:
    count+=1
    news_data_final.append(x[1])

print (count)
docs_test = news_data_final

#SVM
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-03, n_iter 5, random_state=42)),
])
_ = text_clf.fit(news_train_data, news_train_data_target)
predicted = text_clf.predict(docs_test)

print(predicted.size)

fh = open("news_output.txt", 'w')
for item in predicted:
  fh.write("%s\n" % item)
