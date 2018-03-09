# # Project 4: West Nile Virus in the City of Chicago
# 
# Michael Schillawski, 9 March 2018
# 
# Data Science Immersive, General Assembly

# Scoring
# ## 1. Python Setup

# ### 1.1 Import
import sys
import logging
import datetime
from time import gmtime, strftime
sys.stdout=open("train_log.txt","w")

print('starting imports...')

import os
import pickle
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pdp

from haversine import haversine
from joblib import Parallel, delayed


# ## 2. Modeling/Scoring Environment Setup
print('modeling...')

# ### 2.1 Imports
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegressionCV,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

# ### 2.2 Function Definition

def aucroc(probas,y_true,step=0.01):  #,metric='sensitivity',threshold=95
    obs = y_true.values

    sensitivity = []
    specificity = []

    for t in np.arange(0,1,step): #iterate through each step of classification threshold
        
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        
        for i in range(len(y_true)): #iterate through each observation
            predictions = probas[:,1] > t #only predicted class probability

            ##classify each based on whether correctly predicted
            if predictions[i] == 1 and obs[i] == 1:
                TP += 1
            elif predictions[i] == 0 and obs[i] == 1:
                FN += 1
            elif predictions[i] == 1 and obs[i] == 0:
                FP += 1
            elif predictions[i] == 0 and obs[i] == 0:
                TN += 1
        
        #calculate each metric
        sens = TP / (TP + FN)
        spec = TN / (TN + FP)

        #append all metrics to list 
        sensitivity.append(sens)
        specificity.append(1 - spec)

    #graph sens vs spec curve
    plt.rcParams['font.size'] = 14
    plt.plot(specificity,sensitivity)
    plt.plot([0,1],[0,1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('Receiver Operating Characteristic')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

# ### 2.3 Read in models

os.chdir('assets/gradient_boost_pickles')

pickle_jar = [file for file in os.listdir() if '.pickle' in file]

gb = [gb_models,gb_scores,gb_probas,gb_preds,gb_metrics,X_trains,X_tests,y_trains,y_tests]

for p,pickle in enumerate(pickle_jar):
    with open('pickle', 'rb') as file_handle:
        gb[p] = pickle.load(file_handle)

# ## 3. Model Evaluation

