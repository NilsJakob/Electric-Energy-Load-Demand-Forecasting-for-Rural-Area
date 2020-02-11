#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#function to read load data and weather data:
def data_reader(file_name):
    data = pd.read_excel(file_name, parse_dates=True, index_col='Time', usecols=range(2))
    return data

def weather_reader(file_name):
    weather = pd.read_excel(file_name, parse_dates=True, index_col='Time measured')
    return weather

#function for concatenating load data and weather data for training:
def data(file_name_load, file_name_weather):
    train_data = pd.concat([file_name_load, file_name_weather], axis=1)
    return train_data


# In[3]:


load_data = data_reader('Index_Bjønntjønn_2014_2018.xlsx')
weather_data = weather_reader('bo_temp_2014_2018.xlsx')

weather_data = weather_data.interpolate()
Training = data(load_data, weather_data)
#print(Training.head())

#Renaming columns for easier interpreting:
Training = Training.rename(columns={"Total":"Load","Middeltemperatur i 2m høyde (TM)": "Temperature"})
Training.describe()

#Binary series to distuinguish working days from holidays by 1 and 0:
s = pd.date_range('2014-01-01', '2019-01-01', freq='H').to_series()
Training['weekday'] = s.dt.dayofweek
#Training['weekday'] = Training['weekday'].astype(int)
Training['working_days'] = Training['weekday'].replace({6: 1, 5: 1, 4: 1, 3: 0, 2: 0, 1: 0})


# In[4]:


#function to create sliding window based on time shifts:
def time_shifts_func(name, data_hrs, time_shift, regr=False):
    # name = 'DK1'
    # time_shift = 24
    if not regr:
        data_hrs[name + '_t' + '+' + str(time_shift)] = data_hrs[name].shift(time_shift)
    else:
        data_hrs['auto_' + name + '_t' + '+' + str(time_shift)] = (data_hrs[name].shift(time_shift)-data_hrs[name].shift(time_shift+1))
    #print(data_hrs[name].shift(time_shift))
    #data_hours['DK1_t+24'] = data_hours[name].shift(+24)
    #data_hours['DK1_t+168'] = data_hours[name].shift(+168)
    #data_hours['DK1_t-24'] = data_hours[name].shift(-24)
    #return data_hrs
time_shifts_func('Load', Training, 1)
time_shifts_func('Load', Training, 2)   
time_shifts_func('Load', Training, 24)
time_shifts_func('Temperature', Training, 24)
#time_shifts_func('Load - kWh', Training, 168)
#time_shifts_func('Load - kWh', Training,  24, regr=True)
#time_shifts_func('Load - kWh', Training,  1, regr=True)

#Training=Training.dropna()


# In[5]:


Training.head(10)


# In[6]:


def show_plots(data, time_start, time_end=None):
    fig, ax = plt.subplots(figsize=(8,6))
    ax2 = ax.twinx()
    load = data['Load'].loc[time_start:time_end].plot(c='seagreen', label='Load', ax=ax)
    temp = data['Temperature'].loc[time_start:time_end].plot(c='darkorange', label='Temperature', ax=ax2)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.set_ylabel('Load', fontsize=12, fontweight='bold', color='seagreen')
    ax2.set_ylabel('Temperature', fontsize=12, fontweight='bold', color='darkorange')
    fig.tight_layout()
    plt.show()
    return


# In[7]:


# Time-series for 2018
show_plots(Training, '2018')


# In[8]:


# Time-series for June to August 2018
show_plots(Training, '2018-06', '2018-08')


# In[9]:


def engineer_features(dataframe, columns, time_lags=24, drop_nan_rows=True):
    data = dataframe[columns].copy()
    # Features engineering
    for col in data.columns:
        for i in range(1, time_lags+1):
            # Shift data by lag of 1 to time_lags (default: 24) hours
            data[col+'_{:d}h'.format(i)] = data[col].shift(periods=i)  # time-lag
        data[col+'_diff'] = data[col].diff()  # first-difference
        data[col+'_week'] = data[col].shift(periods=24*7)  # previous week
    # Hour-of-day indicators with cyclical transform
    dayhour_ind = data.index.hour
    data['hr_sin'] = np.sin(dayhour_ind*(2.*np.pi/24))
    data['hr_cos'] = np.cos(dayhour_ind*(2.*np.pi/24))
    # Day-of-week indicators with cyclical transform
    weekday_ind = data.index.weekday
    data['week_sin'] = np.sin(weekday_ind*(2.*np.pi/7))
    data['week_cos'] = np.cos(weekday_ind*(2.*np.pi/7))
    # Weekend indicator
    data['weekend'] = np.asarray([0 if ind <= 4 else 1 for ind in weekday_ind])
    # Month indicators with cyclical transform
    month_ind = data.index.month
    data['mnth_sin'] = np.sin((month_ind-1)*(2.*np.pi/12))
    data['mnth_cos'] = np.cos((month_ind-1)*(2.*np.pi/12))
    if drop_nan_rows:
        # Drop rows with NaN values
        data.dropna(inplace=True)
    return data


# In[10]:


data_features = engineer_features(Training, columns=['Load', 'Temperature'])
data_features.head()


# In[11]:


print(data_features.columns)


# In[12]:


#Local holidays including prior knowledge about recreation days
#easter 2014
Training['working_days'].loc['2014-04-14':'2014-04-21']=1
#1st of may 2014
Training['working_days'].loc['2014-05-01']=1
#Pentecost 2014
Training['working_days'].loc['2014-06-07':'2014-06-10']=1
#X-mas 2014
Training['working_days'].loc['2014-12-21':'2014-12-31']=1
#1st of January 2015
Training['working_days'].loc['2015-01-01']
#easter 2015 
Training['working_days'].loc['2015-03-30':'2015-04-06']=1
#1st of may 2015 is on a friday, already coded 
#Training['working_days'].loc['2015-05-01']=1
#Ascension Day 2015 2015-05-14
Training['working_days'].loc['2015-05-14']=1
#Pentecost 2015
Training['working_days'].loc['2014-05-24':'2014-05-25']=1
#X-mas 2015
Training['working_days'].loc['2015-12-23':'2015-12-31']=1
#1st of January 2016
Training['working_days'].loc['2016-01-01']=1
#easter 2016
Training['working_days'].loc['2016-03-21':'2016-03-28']=1
#1st of may 2016 is on a sunday, already coded 
#Training['working_days'].loc['2015-05-01']=1
#Ascension Day 2016 2015-05-16
Training['working_days'].loc['2016-05-05']=1
#Pentecost 2016
Training['working_days'].loc['2016-05-16':'2016-05-17']=1
#X-mas 2016
Training['working_days'].loc['2016-12-26':'2016-12-31']=1
#1st of January 2017
Training['working_days'].loc['2017-01-01']
#easter 2017
Training['working_days'].loc['2017-04-10':'2017-04-17']=1
#1st of may 2017 is on a monday:
Training['working_days'].loc['2017-05-01']=1
#17th of may 2017 is on a wednesday:
Training['working_days'].loc['2017-05-17']=1
#Ascension Day 2017 2017-05-25
Training['working_days'].loc['2017-05-25']=1
#Pentecost 2017
Training['working_days'].loc['2017-06-05']=1
#X-mas 2017
Training['working_days'].loc['2017-12-25':'2017-12-31']=1
#1st of January 2018
Training['working_days'].loc['2018-01-01']
#easter 2018
Training['working_days'].loc['2018-03-26':'2018-04-02']=1
#1st of may 2018 is on a tuesday:
Training['working_days'].loc['2018-05-01']=1
#Ascension Day 2018 2018-05-10
Training['working_days'].loc['2017-05-10']=1
#17th of may 2018 is on a thursday:
Training['working_days'].loc['2017-05-17']=1
#Pentecost 2018
Training['working_days'].loc['2018-05-21']=1
#X-mas 2018
Training['working_days'].loc['2018-12-24':'2018-12-31']=1

