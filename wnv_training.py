
# coding: utf-8

# # Scoring Data
# ### 1.1 Import
import sys
sys.stdout=open("test_log.txt","w")

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import os
import datetime
import pandas_profiling as pdp
from haversine import haversine
from joblib import Parallel, delayed
import multiprocessing

# ## Import Data

#train = pd.read_csv('train.csv')
test_spray = pd.read_csv('spray.csv')
test_weather = pd.read_csv('weather.csv')
testchunks = pd.read_csv('test.csv',chunksize=15000)

for number,test in enumerate(testchunks):

    # ## Data Inspect

    #DataInspect(test)

    #pdp.ProfileReport(test)

    # ## Fix Data
    print('munging...')
    #train['Date'] = pd.to_datetime(train['Date'])
    test_spray['Date'] = pd.to_datetime(test_spray['Date'])
    test_weather['Date'] = pd.to_datetime(test_weather['Date'])
    test['Date'] = pd.to_datetime(test['Date'])

    test_spray['Date'].describe()

    test['Date'].describe()

    test_weather['Date'].describe()

    # ## Drop Duplicates
    # inspect duplicates in spray

    print(test.duplicated().sum())
    test[test.duplicated(keep=False)]

    # inspect duplicates in spray
    print(test_spray.duplicated(keep=False).sum())
    test_spray[test_spray.duplicated(keep=False)][:5]

    test_spray.drop(test_spray[test_spray.duplicated(keep='first')].index,axis=0,inplace=True)

    print(len(test))
    print(len(test_spray))
    print(len(test_weather))

    # ## Fix weather data
    # Changes string T to 0.005, M to 0.0 (11 observations of 2944), and everything else into a float
    def rainy_day(column):
        test_weather[column] = test_weather[column].str.replace('T','0.005')
        test_weather[column] = test_weather[column].str.replace('M','0.0')
        test_weather[column] = test_weather[column].astype(float)

    for col in ['Tavg','PrecipTotal','Depart','WetBulb','SnowFall',
                'StnPressure','SeaLevel','Depth','AvgSpeed','Heat','Cool']:
        rainy_day(col)

    #All Tavg values that were missing (changed to 0 with rainy_day function) are given values from min/max
    test_weather['Tavg'][test_weather.Tavg==0] = (test_weather['Tmin'] + test_weather['Tmax']) / 2
    test_weather['Tavg'].value_counts()

    # ## Make numeric
    print('make numeric...')
    cols = [col for col in test_weather.columns if col not in ('Station','Date')]

    bad_col = []

    for col in cols:
        try:
            test_weather[col] = pd.to_numeric(test_weather[col])
        except:
            bad_col.append(col)
    print(bad_col)

    test_weather.drop('CodeSum',axis=1,inplace=True)
    bad_col.remove('CodeSum')

    bad_data = {}

    for col in bad_col:
        z = []
        for i in test_weather[col]:
            try:
                pd.to_numeric(i)
            except:
                if i in z:
                    pass
                else:
                    z.append(i)
        bad_data[col] = z
    bad_data

    test_weather[bad_col].describe()

    test_weather.drop(bad_col,axis=1,inplace=True)

    # ## Date Engineering

    # need to do something that incorporates date information
    # probably categorical?
    # other options? seasonal, weekly
    # continuous but circular

    # continuous, as day of year
    test["dayofyear"] = test['Date'].dt.dayofyear

    # month
    # train['month'] = train['Date'].dt.month

    # quarter
    # train['quarter'] = train['Date'].dt.quarter

    # ## Dummy species
    # get dummies for mosquito species

    test = pd.get_dummies(test, columns=['Species'])

    # ## Join weather data to trap data

    print(len(test),len(test_weather))

    test_weather.columns

    station1 = test_weather[test_weather['Station']==1].copy()
    station2 = test_weather[test_weather['Station']==2].copy()

    #Station 1: CHICAGO O'HARE INTERNATIONAL AIRPORT Lat: 41.995 Lon: -87.933 Elev: 662 ft. above sea level
    station1['Latitude'] = 41.995
    station1['Longitude'] = -87.9336

    #Station 2: CHICAGO MIDWAY INTL ARPT Lat: 41.786 Lon: -87.752 Elev: 612 ft. above sea level
    station2['Latitude'] = 41.78611
    station2['Longitude'] = -87.75222

    stations = pd.merge(station1,station2,on='Date',suffixes=('_s1','_s2'))
    testtrapsweather = pd.merge(test,stations,on='Date')

    #DataInspect(testtrapsweather)
    print('calculating weather station distance...')
    # ## Calculate point estimates of weather data at trap location
    # calculate distance of traps to weather stations
    dist_1 = np.sqrt(((testtrapsweather['Latitude'] - testtrapsweather['Latitude_s1'])**2 + 
    (testtrapsweather['Longitude'] - testtrapsweather['Longitude_s1'])**2))

    dist_2 = np.sqrt(((testtrapsweather['Latitude'] - testtrapsweather['Latitude_s2'])**2 + 
    (testtrapsweather['Longitude'] - testtrapsweather['Longitude_s2'])**2))

    # calculate distance weights to each trap
    # to weight the weather data by proximity
    total_dist = dist_1 + dist_2
    testtrapsweather['weight_1'] = dist_1 / total_dist
    testtrapsweather['weight_2'] = dist_2 / total_dist

    # Apply distance weights to weather data
    # Inverse weight because the closer station should have the heavier weight

    station1_list = [col for col in testtrapsweather.columns 
                    if '_s1' in col and col not in ('Station_s1','Latitude_s1','Longitude_s1')]
    station2_list = [col for col in testtrapsweather.columns 
                    if '_s2' in col and col not in ('Station_s2','Latitude_s2','Longitude_s2')]

    for col in station1_list:
        testtrapsweather[col] = testtrapsweather['weight_2'] * testtrapsweather[col]
    for col in station2_list:
        testtrapsweather[col] = testtrapsweather['weight_1'] * testtrapsweather[col]

    testtrapsweather.columns

    # we are adding the weighted station1 and station2 weather data
    # and dropping the partial columns

    for col in [col for col in testtrapsweather.columns 
                if 's1' in col and col not in ('Station_s1','Latitude_s1','Longitude_s1')]:
        name = col.replace('_s1','')
        testtrapsweather[name] = testtrapsweather[col] + testtrapsweather[name+'_s2']
        testtrapsweather.drop([col,name+'_s2'],axis=1,inplace=True)

    testtrapsweather.columns

    # Drop station information columns
    col1 = [col for col in testtrapsweather.columns if '_s1' in col]
    col2 = [col for col in testtrapsweather.columns if '_s2' in col]
    cols = col1 + col2

    testtrapsweather.drop(cols,axis=1,inplace=True)

    # ## Spraying data
    # ### Feature Engineering: Binary Flag - Whether Trap affected by Spraying w/in Last 6 months/0.5 mile
    print('calculating spray data...')
    # we calculate the haversine (great circle) distance between the spray intervention and trap site
    # we calculate the time (days) between the spray intervention and trap site
    # together, we create a binary flag for denoting whether the intervention occurred within a recent window
    # relative to the trap observation, both in time and space
    # e.g. within 1 week and within 0.25 miles
    # these we will iterate through our model, and measure the effectiveness of intervention, through the 
    # increase/decrease of WNV probability at our trap sites.

    from haversine import haversine

    print('calculating spray data...')

    def distance_calc(i): 
        
        temp_lat = testtrapsweather.at[i,'Latitude']
        temp_long = testtrapsweather.at[i,'Longitude']

        # calculate distance from traps to spray locations
        dists = []
        if i % 500 == 0:
            print('distance '+str(i))
        #https://stackoverflow.com/questions/43221208/iterate-over-pandas-dataframe-using-itertuples
        for row in test_spray.itertuples(index=True,name=None):        
            dist = haversine((row[3],row[4]),(temp_lat,temp_long),miles=True)
            dists.append(dist)
    
        return dists


    def time_calc(i): 
        if i % 500 == 0:
            print('time '+str(i))        
        # calculate time since spray
        time_since_spray = testtrapsweather.at[i,'Date'] - test_spray['Date']
        time_since_spray = time_since_spray.dt.total_seconds()
        time_since_spray = (((time_since_spray/60)/60)/24)
        
        return time_since_spray

    num_cores = multiprocessing.cpu_count()
    inputs = testtrapsweather.index

    distance_binary = Parallel(n_jobs=num_cores)(delayed(distance_calc)(i) for i in inputs)
    time_binary = Parallel(n_jobs=num_cores)(delayed(time_calc)(i) for i in inputs)
    print('make binaries... done')

    distance_binary = pd.DataFrame(distance_binary)
    time_binary = pd.DataFrame(time_binary)

    time_binary.reset_index(inplace=True)
    time_binary.drop('index',axis=1,inplace=True)

    # if observation took place before spray, zero out time
    # else return elapsed time between spray and observation

    print('negating sprays after traps...')
    
    for col in time_binary.columns:
        time_binary[col] = time_binary[col].map(lambda x: 0 if x < 0 else x)
    print('done')
       
    # https://chrisalbon.com/python/data_wrangling/pandas_rename_multiple_columns/
    time_binary.columns = distance_binary.columns

    time_binary_backup = time_binary.copy()
    distance_binary_backup = distance_binary.copy()

    time_tp = time_binary.transpose()
    distance_tp = distance_binary.transpose()

    time_tp = time_binary.transpose()
    distance_tp = distance_binary.transpose()

    def CalculateDistance(i):
        distances = []
        if i % 500 == 0:
            print('evaluating time binaries '+str(i))
        d = i
        cols = distance_tp[[d]]

        if len(cols[np.logical_and(cols[d] <= 0.5,cols[d] > 0)]) > 0:
            distances.append(1)
        else:
            distances.append(0)
        if len(cols[np.logical_and(cols[d] <= 1,cols[d] > 0)]) > 0:
            distances.append(1)
        else:
            distances.append(0)
        if len(cols[np.logical_and(cols[d] <= 5,cols[d] > 0)]) > 0:
            distances.append(1)
        else:
            distances.append(0)
            
        if len(cols[np.logical_and(cols[d] <= 0.5,cols[d] > 0)]) > 0:
            distances.append(1)
        else:
            distances.append(0)
        if len(cols[np.logical_and(cols[d] <= 1,cols[d] > 0)]) > 0:
            distances.append(1)
        else:
            distances.append(0)
        if len(cols[np.logical_and(cols[d] <= 5,cols[d] > 0)]) > 0:
            distances.append(1)
        else:
            distances.append(0)

        if len(cols[np.logical_and(cols[d] <= 0.5,cols[d] > 0)]) > 0:
            distances.append(1)
        else:
            distances.append(0)
        if len(cols[np.logical_and(cols[d] <= 1,cols[d] > 0)]) > 0:
            distances.append(1)
        else:
            distances.append(0)
        if len(cols[np.logical_and(cols[d] <= 5,cols[d] > 0)]) > 0:
            distances.append(1)
        else:
            distances.append(0)
        return distances

    def CalculateTime(i):
        times = []
        if i % 500 == 0:
            print('evaluating time binaries '+str(i))

        t = i

        cols = time_tp[[t]]
        
        if len(cols[np.logical_and(cols[t] <= 7,cols[t] > 0)]) > 0:
            times.append(1)
        else:
            times.append(0)
        if len(cols[np.logical_and(cols[t] <= 7,cols[t] > 0)]) > 0:
            times.append(1)
        else:
            times.append(0)
        if len(cols[np.logical_and(cols[t] <= 7,cols[t] > 0)]) > 0:
            times.append(1)
        else:
            times.append(0)
            
        if len(cols[np.logical_and(cols[t] <= 30,cols[t] > 0)]) > 0:
            times.append(1)
        else:
            times.append(0)
        if len(cols[np.logical_and(cols[t] <= 30,cols[t] > 0)]) > 0:
            times.append(1)
        else:
            times.append(0)
        if len(cols[np.logical_and(cols[t] <= 30,cols[t] > 0)]) > 0:
            times.append(1)
        else:
            times.append(0)

        if len(cols[np.logical_and(cols[t] <= 90,cols[t] > 0)]) > 0:
            times.append(1)
        else:
            times.append(0)
        if len(cols[np.logical_and(cols[t] <= 90,cols[t] > 0)]) > 0:
            times.append(1)
        else:
            times.append(0)
        if len(cols[np.logical_and(cols[t] <= 90,cols[t] > 0)]) > 0:
            times.append(1)
        else:
            times.append(0)
        return times
        
    print('binary bonanza...')
    inputs = testtrapsweather.index

    timevalues = Parallel(n_jobs=num_cores)(delayed(CalculateTime)(i) for i in inputs)
    distancevalues = Parallel(n_jobs=num_cores)(delayed(CalculateDistance)(i) for i in inputs)

    timevalues = pd.DataFrame(timevalues)
    distancevalues = pd.DataFrame(distancevalues)
    tv = timevalues.shape[1]
    binary = pd.merge(distancevalues,timevalues,how='inner',left_index=True,right_index=True,suffixes=('_d','_t'))

    for c in range(tv):
        binary[c] = binary[str(c)+'_d'] + binary[str(c)+'_t']

    binary.rename(columns={0:'1week_halfmile',1:'1week_1mile',2:'1week_5mile',
                        3:'1month_halfmile',4:'1month_1mile',5:'1month_5mile',
                        6:'1quarter_halfmile',7:'1quarter_1mile',8:'1quarter_5mile'},inplace=True)

    timecols = [col for col in binary.columns if '_t' in col]
    distcols = [col for col in binary.columns if '_d' in col]
    dropcols = timecols + distcols

    binary.drop(dropcols,axis=1,inplace=True)

    for col in binary.columns:
        binary[col] = binary[col].apply(lambda x: 1 if x == 2 else 0)

    binary.to_csv('spraytest_data_binary'+str(number)+'.csv')

    data = pd.merge(testtrapsweather,binary,left_index=True,right_index=True)

    print('exporting...')
    # ## Export cleaned test data
    data.to_csv('chunk'+str(number)+'.csv')

    # # ## Scoring
    # # ## Export scored data

    # ids = clean_test_data['Id']

    # predictions = pd.DataFrame(gb_probas[:,1],index=ids)
    # predictions.rename(columns={0:'WnvPresent'},inplace=True)

    # #path = '/Users/mjschillawski/Google Drive/Data/generalassembly/projects/west_nile_virus/assets/output'
    # predictions.to_csv('predictions_'+model_name+'.csv')

sys.stdout.close()
