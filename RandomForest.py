#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 14:38:21 2022

@author: Nana Effiansah Asmah
"""

import pickle
import sklearn
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
import time
import tkinter
from tkinter import filedialog
from numpy import loadtxt
import matplotlib.pyplot as plt


# load dataset
print('\n \n SELECT TRAINING DATASET')
# displays time of operation
print(time.strftime("%b %d %y %H:%M", time.localtime()))
time.sleep(2)

# loading the training dataset
training = filedialog.askopenfilename(initialdir = '/Desktop',
                                     title = ' SELECT THE TRAINING DATASET')
Train = loadtxt(training, delimiter = ',')

print('\n \n SELECT VALIDATION DATASET')
print(time.strftime("%b %d %y %H:%M", time.localtime()))
time.sleep(3)
# loading the validation dataset
validation = filedialog.askopenfilename(initialdir = '/Desktop',
                                     title = ' SELECT THE VALIDATION DATASET')
Valid = loadtxt(validation, delimiter = ',')

# spits dataset into input and output variables
x_train = Train[:, 0:777]
y_train = Train[:, 777]
x_val = Valid[:, 0:777]
y_val = Valid[:,777]



print('\n \n SELECT TESTING DATASET')
print(time.strftime("%b %d %y %H:%M", time.localtime()))
time.sleep(4)
#loading the testing dataset
testing=filedialog.askopenfilename(initialdir = '/Desktop',      
 title = '                         SELECT TEST DATASET     ')
Test = loadtxt(testing,delimiter=',')
print('    RUNNING PROGRAM\n\n')
time.sleep(3)
x_test = Test[:,0:777]
y_test= Test[:,777]


# create random forest classifer
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(x_train,y_train)
predictions = classifier.predict(x_test)



#Predict probabilities for test set
yhat_probs = classifier.predict(x_test)
#Predict crisp classes for test set
yhat_classes = classifier.predict(x_test)


#Accuracy: (tp + tn) / (p + y_test)
accuracy = accuracy_score(y_test, predictions)
print('Accuracy: %f' % accuracy)

#Balanced accuracy
balanced_accuracy = balanced_accuracy_score(y_test, predictions)
print('Balanced accuracy: %f' % balanced_accuracy)

#Precision tp / (tp + fp)
precision = precision_score(y_test, predictions)
print('Precision: %f' % precision)

#Recall: tp / (tp + fn)
recall = recall_score(y_test, predictions)
print('Recall: %f' % recall)

#Matthew's correlation coefficient
mat = matthews_corrcoef(y_test, predictions)
print('MCC: %f' % mat)

#f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, predictions)
print('F1 score: %f' % f1)

#Kappa
kappa = cohen_kappa_score(y_test, predictions)
print('Cohens kappa: %f' % kappa)

#ROC AUC
auc = roc_auc_score(y_test, predictions)
print('ROC AUC: %f' % auc)

#Confusion matrix
matrix = confusion_matrix(y_test, predictions)
print(matrix)
print(classification_report(y_test, predictions))


# save the model to disk
filename = "RF_model.sav"
pickle.dump(classifier, open(filename, 'wb'))