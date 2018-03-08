
# # Project 4: West Nile Virus in the City of Chicago
# 
# Michael Schillawski, 9 March 2018
# 
# Data Science Immersive, General Assembly

# ## 1. Setup

# ### 1.1 Import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import pandas_profiling as pdp
from haversine import haversine
from joblib import Parallel, delayed
import multiprocessing

# ### 1.2 Gather Data

path = os.getcwd()

if path != '/Users/mjschillawski/Google Drive/Data/generalassembly/projects/west_nile_virus':
    path = '/Users/mjschillawski/Google Drive/Data/generalassembly/projects/west_nile_virus'
else:
    pass
path = path + '/assets/input'
os.chdir(path)

ls

train = pd.read_csv('train.csv')
spray = pd.read_csv('spray.csv')
weather = pd.read_csv('weather.csv')

# ## 2. Data Cleaning
# ### 2.1 Formatting & DeDuping
# #### 2.1.1 Inspecting

def DataInspect(dataframe):
    '''Original function (previously called eda) created by Ritika Bhasker
       Good first step when starting any project. Provides overview of
       dataset including missing values, duplicates and types.
       Takes a Pandas dataframe as the argument.
       Modified by Michael Schillawski based on his preferences.'''
    print("Dataframe Shape:", dataframe.shape,"\n")
    print("Duplicate Rows:", dataframe.duplicated().sum(),"\n") #Added this
    print("Dataframe Types \n\n", dataframe.dtypes,"\n")    
    print("Missing Values \n\n", dataframe.isnull().sum(),"\n")
    print("Dataframe Describe \n\n", dataframe.describe(include='all'),"\n")


    print('Unique Values by Variable')
    for item in dataframe:
        print(item,':',dataframe[item].nunique())       

DataInspect(train)
DataInspect(spray)
DataInspect(weather)
pdp.ProfileReport(train)
pdp.ProfileReport(spray)
pdp.ProfileReport(weather)

# ### 2.2 Munging
# #### 2.2.1 Fix dates
train['Date'] = pd.to_datetime(train['Date'])
spray['Date'] = pd.to_datetime(spray['Date'])
weather['Date'] = pd.to_datetime(weather['Date'])
#test['Date'] = pd.to_datetime(test['Date'])

spray['Date'].describe()
train['Date'].describe()
weather['Date'].describe()

# #### 2.1.2 Drop Duplicates
# inspect duplicates in spray
train[train.duplicated(keep=False)][:5]
train.drop(train[train.duplicated(keep='first')].index,axis=0,inplace=True)

# inspect duplicates in spray
spray[spray.duplicated(keep=False)][:5]
spray.drop(spray[spray.duplicated(keep='first')].index,axis=0,inplace=True)

print(len(train))
print(len(spray))
print(len(weather))

mosq = [col for col in train.columns if 'Species_' in col]
train[mosq].head(3)

# #### 2.1.3 Fix weather data
# Changes string T to 0.005, M to 0.0 (11 observations of 2944), and everything else into a float
def rainy_day(column):
    weather[column] = weather[column].str.replace('T','0.005')
    weather[column] = weather[column].str.replace('M','0.0')
    weather[column] = weather[column].astype(float)

for col in ['Tavg','PrecipTotal','Depart','WetBulb','SnowFall',
            'StnPressure','SeaLevel','Depth','AvgSpeed','Heat','Cool']:
    rainy_day(col)

#All Tavg values that were missing (changed to 0 with rainy_day function) are given values from min/max
weather['Tavg'][weather.Tavg==0] = (weather['Tmin'] + weather['Tmax']) / 2
weather['Tavg'].value_counts()

# #### 2.1.4 Make numeric
cols = [col for col in weather.columns if col not in ('Station','Date')]

bad_col = []

for col in cols:
    try:
        weather[col] = pd.to_numeric(weather[col])
    except:
        bad_col.append(col)
print(bad_col)

weather.drop('CodeSum',axis=1,inplace=True)
bad_col.remove('CodeSum')

bad_data = {}

for col in bad_col:
    z = []
    for i in weather[col]:
        try:
            pd.to_numeric(i)
        except:
            if i in z:
                pass
            else:
                z.append(i)
    bad_data[col] = z
bad_data

weather[bad_col].describe()
weather.drop(bad_col,axis=1,inplace=True)

# #### 2.1.5 Aggregate traps
# aggregate trap observations by date and species
# observations are split when n_mosquitoes > 50
# rebuild trap observations
# inspect whether both observations are flagged as 1 / 0 or not
# whether to replace 1 across both if true

print(train.duplicated(['Date','Species','Trap']).sum())
train[train.duplicated(['Date','Species','Trap'],keep=False)].sort_values(['Date','Trap','Species'])['WnvPresent'].value_counts()
train['WnvPresent'].value_counts(())

train = train.groupby(['Date','Address','Species','Block','Street',
              'Trap','AddressNumberAndStreet','Latitude','Longitude','AddressAccuracy']).sum().reset_index()
train['WnvPresent'] = train['WnvPresent'].map(lambda x: 1 if x >= 1 else 0)
train['WnvPresent'].value_counts()

len(train)

# #### 2.1.6 Date Engineering
# need to do something that incorporates date information
# probably categorical?
# other options? seasonal, weekly
# continuous but circular

# continuous, as day of year
train["dayofyear"] = train['Date'].dt.dayofyear

# month
# train['month'] = train['Date'].dt.month

# quarter
# train['quarter'] = train['Date'].dt.quarter

# #### 2.1.7 Dummy species
# get dummies for mosquito species
train = pd.get_dummies(train, columns=['Species'])

# #### Map WNV outbreaks
# mapdata = np.loadtxt("./mapdata_copyright_openstreetmap_contributors.txt")
# traps = pd.read_csv('./train.csv')[['Date', 'Trap','Longitude', 'Latitude', 'WnvPresent']]

# aspect = mapdata.shape[0] * 1.0 / mapdata.shape[1]
# lon_lat_box = (-88, -87.5, 41.6, 42.1)

# plt.figure(figsize=(10,14))
# plt.imshow(mapdata,
#           cmap=plt.get_cmap('gray'),
#           extent=lon_lat_box,
#           aspect=aspect)

# traps1 = traps[traps.WnvPresent==1]
# traps0 = traps[traps.WnvPresent==0]

# traps1 = traps1[['Longitude', 'Latitude']].drop_duplicates().values
# traps0 = traps0[['Longitude', 'Latitude']].drop_duplicates().values

# plt.scatter(traps0[:,0], traps0[:,1], color='pink', marker='*', alpha=1, label='Wnv = No');
# plt.scatter(traps1[:,0], traps1[:,1], color='purple', marker='*', alpha=1, label='Wnv = Yes');
# plt.legend();

# plt.savefig('trap_map.png');

# ## 3. EDA
# ### 3.1 Join Weather Data to Trap Data
print(len(train),len(weather))

weather.columns

station1 = weather[weather['Station']==1].copy()
station2 = weather[weather['Station']==2].copy()

#Station 1: CHICAGO O'HARE INTERNATIONAL AIRPORT Lat: 41.995 Lon: -87.933 Elev: 662 ft. above sea level
station1['Latitude'] = 41.995
station1['Longitude'] = -87.9336

#Station 2: CHICAGO MIDWAY INTL ARPT Lat: 41.786 Lon: -87.752 Elev: 612 ft. above sea level
station2['Latitude'] = 41.78611
station2['Longitude'] = -87.75222

stations = pd.merge(station1,station2,on='Date',suffixes=('_s1','_s2'))
traps_weather = pd.merge(train,stations,on='Date')

DataInspect(traps_weather)

# ### 3.2 Calculate point estimates of weather data at trap location
# calculate distance of traps to weather stations
dist_1 = np.sqrt(((traps_weather['Latitude'] - traps_weather['Latitude_s1'])**2 + 
 (traps_weather['Longitude'] - traps_weather['Longitude_s1'])**2))

dist_2 = np.sqrt(((traps_weather['Latitude'] - traps_weather['Latitude_s2'])**2 + 
 (traps_weather['Longitude'] - traps_weather['Longitude_s2'])**2))

# calculate distance weights to each trap
# to weight the weather data by proximity
total_dist = dist_1 + dist_2
traps_weather['weight_1'] = dist_1 / total_dist
traps_weather['weight_2'] = dist_2 / total_dist

# Apply distance weights to weather data
# Inverse weight because the closer station should have the heavier weight
station1_list = [col for col in traps_weather.columns 
                if '_s1' in col and col not in ('Station_s1','Latitude_s1','Longitude_s1')]
station2_list = [col for col in traps_weather.columns 
                 if '_s2' in col and col not in ('Station_s2','Latitude_s2','Longitude_s2')]

for col in station1_list:
    traps_weather[col] = traps_weather['weight_2'] * traps_weather[col]
for col in station2_list:
    traps_weather[col] = traps_weather['weight_1'] * traps_weather[col]

traps_weather.columns

# we are adding the weighted station1 and station2 weather data
# and dropping the partial columns

for col in [col for col in traps_weather.columns 
            if 's1' in col and col not in ('Station_s1','Latitude_s1','Longitude_s1')]:
    name = col.replace('_s1','')
    traps_weather[name] = traps_weather[col] + traps_weather[name+'_s2']
    traps_weather.drop([col,name+'_s2'],axis=1,inplace=True)

traps_weather.columns

# Drop station information columns
col1 = [col for col in traps_weather.columns if '_s1' in col]
col2 = [col for col in traps_weather.columns if '_s2' in col]
cols = col1 + col2

traps_weather.drop(cols,axis=1,inplace=True)

# ### 3.4 Spraying Data
# #### Feature Engineering: Calculate Spray in Time and Space from Each Trap Observation
# len(spray)
# spray.head(3)

# #WARNING: THIS IS A COMPUTATIONALLY INTENSIVE CELL

# # the idea here is that the targeted intervention has an effect that decays in two dimensions, time and distance
# # the reference location that we care about is the trap
# # so we evaluate spraying by how far the trap is from the where the spraying occurs AND
# # we evaluate how long before the trap observation did the spraying occur
# # so we calculate the deltas for every spraying against every trap
# # if spraying occurred after the observation, we zero out these observations

# # should we cross-multiply the distance and time? spraying that is 
# # close in both time and distance should be privileged

# distance = []
# time = []

# for i in traps_weather.index:
#     temp_lat = traps_weather.at[i,'Latitude']
#     temp_long = traps_weather.at[i,'Longitude']

#     # calculate distance from traps to spray locations
#     dist = np.sqrt((spray['Latitude'] - temp_lat)**2 + (spray['Longitude'] - temp_long)**2)
#     distance.append(dist)

#     # calculate time since spray
#     time_since_spray = traps_weather.at[i,'Date'] - spray['Date']
#     time_since_spray = time_since_spray.dt.total_seconds()
#     time_since_spray = (((time_since_spray/60)/60)/24)
#     time.append(time_since_spray)

# distance = pd.DataFrame(distance)
# time = pd.DataFrame(time)

# time.reset_index(inplace=True)
# time.drop('index',axis=1,inplace=True)

# backup = time.copy()

# # if observation took place before spray, zero out time
# # else return elapsed time between spray and observation

# for col in time.columns:
#     time[col] = time[col].map(lambda x: 0 if x < 0 else x)

# time.head()

# # join spray distances from traps and spray elapsed time from traps
# # we now calculated the distance (time and space) between mosquito spraying and each trap observation
# # in the 'data' df, each spray event has a _d (distance from trap) and _t (time from trap) column pair

# # data = pd.merge(distance,time,how='inner',left_index=True,right_index=True,suffixes=('_d','_t'))
# # data.shape

# # # join traps_weather with spraying

# # data = pd.merge(traps_weather,data,how='inner',left_index=True,right_index=True)
# # data.shape

# # multiply elapsed time by spray distance from trap for each spray event
# # this creates 1 quantity per spray event, and 
# # incorporates both time and distance, the two dimensions spraying decays in

# spray_data = [time[i] * distance[i] for i in time.columns]
# spray_data = pd.DataFrame(spray_data).transpose()

# data = pd.merge(traps_weather,spray_data,how='inner',left_index=True,right_index=True)
# data.shape


# #### Feature Engineering: Binary Flag - Whether Trap affected by Spraying w/in Last 6 months/0.5 mile
# we calculate the haversine (great circle) distance between the spray intervention and trap site
# we calculate the time (days) between the spray intervention and trap site
# together, we create a binary flag for denoting whether the intervention occurred within a recent window
# relative to the trap observation, both in time and space
# e.g. within 1 week and within 0.25 miles
# these we will iterate through our model, and measure the effectiveness of intervention, through the 
# increase/decrease of WNV probability at our trap sites.

from haversine import haversine

print('starting...')

def distance_calc(i): 
    
    temp_lat = traps_weather.at[i,'Latitude']
    temp_long = traps_weather.at[i,'Longitude']

    # calculate distance from traps to spray locations
    dists = []
    if i % 500 == 0:
        print(i)
    for s in spray.index:
        dist = haversine((test_spray.at[s,'Latitude'],spray.at[s,'Longitude']),(temp_lat,temp_long),miles=True)
        dists.append(dist)
    
    return dists

def time_calc(i): 
        
    # calculate time since spray
    time_since_spray = traps_weather.at[i,'Date'] - spray['Date']
    time_since_spray = time_since_spray.dt.total_seconds()
    time_since_spray = (((time_since_spray/60)/60)/24)
    
    return time_since_spray

num_cores = multiprocessing.cpu_count()
inputs = testtrapsweather.index

distance_binary = Parallel(n_jobs=num_cores)(delayed(distance_calc)(i) for i in inputs)
time_binary = Parallel(n_jobs=num_cores)(delayed(time_calc)(i) for i in inputs)

distance_binary = pd.DataFrame(distance_binary)
time_binary = pd.DataFrame(time_binary)

time_binary.reset_index(inplace=True)
time_binary.drop('index',axis=1,inplace=True)

# if observation took place before spray, zero out time
# else return elapsed time between spray and observation

for col in time_binary.columns:
    time_binary[col] = time_binary[col].map(lambda x: 0 if x < 0 else x)
    
# https://chrisalbon.com/python/data_wrangling/pandas_rename_multiple_columns/
time_binary.columns = distance_binary.columns

time_binary_backup = time_binary.copy()
distance_binary_backup = distance_binary.copy()

time_tp = time_binary.transpose()
distance_tp = distance_binary.transpose()

binary = pd.merge(distance_tp,time_tp,how='inner',left_index=True,right_index=True,suffixes=('_d','_t'))
binary.shape

def CalculateBinary(i):
    observations = []
    if i % 500 == 0:
        print(i)

    d = str(i) + '_d'
    t = str(i) + '_t'
    
    if len(binary[np.logical_and(np.logical_and(binary[d] <= 0.5,binary[d] > 0),
       np.logical_and(binary[t] <= 7,binary[t] > 0))]) > 0:
        observations.append(1)
    else:
        observations.append(0)
    if len(binary[np.logical_and(np.logical_and(binary[d] <= 1,binary[d] > 0),
           np.logical_and(binary[t] <= 7,binary[t] > 0))]) > 0:
        observations.append(1)
    else:
        observations.append(0)
    if len(binary[np.logical_and(np.logical_and(binary[d] <= 5,binary[d] > 0),
       np.logical_and(binary[t] <= 7,binary[t] > 0))]) > 0:
        observations.append(1)
    else:
        observations.append(0)
        
    if len(binary[np.logical_and(np.logical_and(binary[d] <= 0.5,binary[d] > 0),
           np.logical_and(binary[t] <= 30,binary[t] > 0))]) > 0:
        observations.append(1)
    else:
        observations.append(0)
    if len(binary[np.logical_and(np.logical_and(binary[d] <= 1,binary[d] > 0),
           np.logical_and(binary[t] <= 30,binary[t] > 0))]) > 0:
        observations.append(1)
    else:
        observations.append(0)
    if len(binary[np.logical_and(np.logical_and(binary[d] <= 5,binary[d] > 0),
           np.logical_and(binary[t] <= 30,binary[t] > 0))]) > 0:
        observations.append(1)
    else:
        observations.append(0)

    if len(binary[np.logical_and(np.logical_and(binary[d] <= 0.5,binary[d] > 0),
           np.logical_and(binary[t] <= 90,binary[t] > 0))]) > 0:
        observations.append(1)
    else:
        observations.append(0)
    if len(binary[np.logical_and(np.logical_and(binary[d] <= 1,binary[d] > 0),
           np.logical_and(binary[t] <= 90,binary[t] > 0))]) > 0:
        observations.append(1)
    else:
        observations.append(0)
    if len(binary[np.logical_and(np.logical_and(binary[d] <= 5,binary[d] > 0),
           np.logical_and(binary[t] <= 90,binary[t] > 0))]) > 0:
        observations.append(1)
    else:
        observations.append(0)
    return observations

inputs = traps_weather.index
values = Parallel(n_jobs=num_cores)(delayed(CalculateBinary)(i) for i in inputs)

values = pd.DataFrame(values)

values.to_csv('spray_data_binary.csv')

spray_col = ['1week_halfmile','1week_1mile','1week_5mile',
            '1month_halfmile','1month_1mile','1month_5mile',
            '1quarter_halfmile','1quarter_1mile','1quarter_5mile']
values.columns = spray_col

data = pd.merge(traps_weather,values,left_index=True,right_index=True)

# ### 3.5 Export clean data
data.to_csv('cleaned_data.csv')

# ### 3.6 Train/Test/Split
path = '/Users/mjschillawski/Google Drive/Data/generalassembly/projects/west_nile_virus/assets/input'
data = pd.read_csv(path+'/cleaned_data.csv',index_col=0)
data['Date'] = pd.to_datetime(data['Date'])

data.columns[0:45]
# create a list of training data removing all but 1 of the dummy variables
# this will allow us to gridsearch over different dummies
# find the best aggregation and decay of spray data

sprays = ['1week_halfmile','1week_1mile', '1week_5mile', '1month_halfmile', '1month_1mile',
       '1month_5mile', '1quarter_halfmile', '1quarter_1mile','1quarter_5mile']

Xsprays = []

for s in sprays:
    spray_var = ['1week_halfmile','1week_1mile', '1week_5mile', '1month_halfmile', '1month_1mile',
       '1month_5mile', '1quarter_halfmile', '1quarter_1mile','1quarter_5mile']
    
    X_spray = data.drop(['Address','Block','Street','Trap',
           'AddressNumberAndStreet','AddressAccuracy','weight_1',
          'weight_2','Date','NumMosquitos'],axis=1)
    spray_var.remove(s)
    X_spray = X_spray.drop(spray_var,axis=1)
    Xsprays.append(X_spray)

# add no spray data
spray_var = ['1week_halfmile','1week_1mile', '1week_5mile', '1month_halfmile', '1month_1mile',
   '1month_5mile', '1quarter_halfmile', '1quarter_1mile','1quarter_5mile']
X_spray = data.drop(['Address','Block','Street','Trap',
       'AddressNumberAndStreet','AddressAccuracy','weight_1',
      'weight_2','Date','NumMosquitos'],axis=1)
X_spray = X_spray.drop(spray_var,axis=1)
Xsprays.append(X_spray)

from sklearn.model_selection import train_test_split

X_trains = []
X_tests = []
y_trains = []
y_tests = []

for i in Xsprays:
    X = i.drop('WnvPresent',axis=1)
    Y = i['WnvPresent']

    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=20180309,shuffle=True)
    
    X_trains.append(X_train)
    X_tests.append(X_test)
    y_trains.append(y_train)
    y_tests.append(y_test)   

for i in range(len(Xsprays)):
    print(X_trains[i].shape)
    print(X_tests[i].shape)
    print(y_trains[i].shape)
    print(y_tests[i].shape)

from sklearn.model_selection import train_test_split

Xns = data.drop('WnvPresent',axis=1)
Xns = Xns[['dayofyear',        'Species_CULEX ERRATICUS',
                'Species_CULEX PIPIENS', 'Species_CULEX PIPIENS/RESTUANS',
               'Species_CULEX RESTUANS',       'Species_CULEX SALINARIUS',
               'Species_CULEX TARSALIS',        'Species_CULEX TERRITANS',
                                 'Tmax',                           'Tmin',
                                 'Tavg',                         'Depart',
                             'DewPoint',                        'WetBulb',
                                 'Heat',                           'Cool',
                                'Depth',                       'SnowFall',
                          'PrecipTotal',                    'StnPressure',
                             'SeaLevel',                    'ResultSpeed',
                            'ResultDir',                       'AvgSpeed',]]
Yns = data['WnvPresent']

Xns_train,Xns_test,yns_train,yns_test = train_test_split(Xns,Yns,test_size=0.3,random_state=20180309,shuffle=True)

print('T/T/S report for {}'.format('Data without Spraying'))
print('overall shape: {} \n'.format(data.shape))
print('X shape: {}'.format(Xns.shape))
print('X_train shape: {}'.format(Xns_train.shape))
print('X_test shape: {} \n'.format(Xns_test.shape))
print('Y shape: {}'.format(Yns.shape))
print('Y_train shape: {}'.format(yns_train.shape))
print('Y_test shape: {}'.format(yns_test.shape))


# ## 4. Modeling

# We need to predict when, where, and among which species West Nile Virus will occur
# In which traps will we observe West Nile Virus?
# 
# How to define outcome variable?
# 
# Outcome: WNV 1/0
# 
# Variable selection:
# - weather
# - spray
# - species
# - location-based

# ### 4.1 Model Imports
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


# ### Gradient Boosting (No Spray)
# 
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

gradboost = GradientBoostingClassifier(n_estimators=800,
                                       verbose=1,
                                       random_state=20180309)

print('Fitting...')
gradboost.fit(Xns_train,yns_train)

print('Scoring...')
gb_scores = gradboost.score(Xns_train,yns_train)

print(gb_scores)
print('accuracy score: {}'.format(np.mean(gb_scores)))

gb_preds = gradboost.predict(Xns_test)
gb_probas = gradboost.predict_proba(Xns_test)

tn, fp, fn, tp = confusion_matrix(yns_test,gb_preds).ravel()
print('tn,fp,fn,tp')
print(tn,fp,fn,tp)

print('accuracy score: {}:'.format(accuracy_score(yns_test, gb_preds)))
print('precision score: {}'.format(precision_score(yns_test,gb_preds)))
print('recall score: {}'.format(accuracy_score(yns_test,gb_preds)))
print(classification_report(yns_test,gb_preds))

print(roc_auc_score(yns_test,gb_probas[:,1]))

aucroc(gb_probas,yns_test)


# ### Gradient Boosting (Spray Data GridSearch)
# 
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

# source for hyperparameter grid:
# https://datascience.stackexchange.com/questions/14377/
# tuning-gradient-boosted-classifiers-hyperparametrs-and-balancing-it

gb_models = {}
gb_scores = {}
gb_probas = {}
gb_preds = {}
gb_metrics = {}

for i in range(len(Xsprays)):
    print('Iteration: {}'.format(i))
    print('Spray Type: {}'.format(Xsprays[i].columns[-1]))
    
    gb_grid_params = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
              'max_depth': [4, 6, 8],
              'min_samples_leaf': [20,50,100,150],
              'max_features': [1.0, 0.3, 0.1] 
              }

    gradboost = GradientBoostingClassifier(n_estimators=1000,
                                       verbose=1,
                                       random_state=20180309) 

    gb_gscv = GridSearchCV(gradboost,
                                   gb_grid_params,
                                   cv=4,
                                   scoring='roc_auc',
                                   verbose = 3, 
                                   n_jobs=-1)
    print('Fitting...')
    gb_gscv.fit(X_trains[i],y_trains[i])

    print('Scoring...')
    gb_score = gb_gscv.score(X_trains[i],y_trains[i])
    gb_scores[i] = gb_score
    
    print(gb_score)
    print('AUCROC score: {}'.format(np.mean(gb_score)))

    gb_pred = gb_gscv.best_estimator_.predict(X_tests[i])
    gb_preds[i] = gb_pred
    
    gb_proba = gb_gscv.best_estimator_.predict_proba(X_tests[i])
    gb_probas[i] = gb_proba
    
    tn, fp, fn, tp = confusion_matrix(y_tests[i],gb_pred).ravel()
    print('tn,fp,fn,tp')
    print(tn,fp,fn,tp)

    print('accuracy score: {}:'.format(accuracy_score(y_tests[i], gb_pred)))
    print('precision score: {}'.format(precision_score(y_tests[i],gb_pred)))
    print('recall score: {}'.format(accuracy_score(y_tests[i],gb_pred)))
    print(classification_report(y_tests[i],gb_pred))
    
    gb_models[i] = gb_gscv.best_estimator_
    gb_metrics[i] = (accuracy_score(y_tests[i], gb_pred),
                     precision_score(y_tests[i],gb_pred),
                    classification_report(y_tests[i],gb_pred))

for i in range(len(Xsprays)):
    print(roc_auc_score(y_tests[i],gb_probas[i][:,1]))

    aucroc(gb_probas[i],y_tests[i])
    plt.plot()

import pickle
    
# Create a list of files that have '.pickle' in the name:
# (This could be '.csv', or whatever)
#os.chdir('/Users/mjschillawski/Google Drive/Data/generalassembly/projects/west_nile_virus/assets/pickle_jar')
#pickle_jar = [file for file in os.listdir() if '.pickle' in file]

# # Now I just save my next file with a new number, 1 higher:
# df.to_pickle(str(max_pickle_num + 1) + '.pickle')

gb = [gb_models,gb_scores,gb_probas,gb_preds,gb_metrics,X_trains,X_tests,y_trains,y_tests]

for i,val in enumerate(gb):
    with open(str(i)+'.pickle','wb') as file_handle:
        pickle.dump(val,file_handle,protocol=0)
