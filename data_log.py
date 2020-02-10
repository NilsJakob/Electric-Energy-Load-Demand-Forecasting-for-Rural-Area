#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#function to read load data and weather data:
def data_reader(file_name):
    data = pd.read_excel(file_name, parse_dates=True, index_col='Time', usecols=range(2))
    return data

load_data = data_reader('C:/Users/nilsjj12/Documents/Present/Data/Index_Bjønntjønn_2014_2018.xlsx')

def weather_reader(file_name):
    weather = pd.read_excel(file_name, parse_dates=True, index_col='Time measured')
    return weather

weather_data = weather_reader('C:/Users/nilsjj12/Documents/Present/Data/bo_temp_2014_2018.xlsx')

#function for concatenating load data and weather data for training:
def data(file_name_load, file_name_weather):
    train_data = pd.concat([file_name_load, file_name_weather], axis=1)
    return train_data

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

Training.head()

