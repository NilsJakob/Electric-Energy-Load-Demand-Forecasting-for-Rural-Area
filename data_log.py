#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


Training.head(10)


# In[ ]:


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


# In[ ]:


# Time-series for 2018
show_plots(Training, '2018')


# In[ ]:


# Time-series for June to August 2018
show_plots(Training, '2018-06', '2018-08')


# In[ ]:




