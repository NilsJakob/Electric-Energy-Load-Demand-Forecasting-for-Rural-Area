#!/usr/bin/env python
# coding: utf-8

# ## Electric Energy Load Demand Forecasting for Rural Area

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


# function to read load data 
def data_reader(file_name):
    data = pd.read_excel(file_name, parse_dates=True, index_col='Time', usecols=range(2))
    return data

# function to read weather data
def weather_reader(file_name):
    weather = pd.read_excel(file_name, parse_dates=True, index_col='Time measured')
    return weather

# function for concatenating load data and weather data for training
def concat_data(file_name_load, file_name_weather):
    train_data = pd.concat([file_name_load, file_name_weather], axis=1)
    return train_data


# In[ ]:


# Load weather & load time-series data
load_data = data_reader('Index_Bjønntjønn_2014_2018.xlsx')
weather_data = weather_reader('bo_temp_2014_2018.xlsx')
weather_data = weather_data.interpolate()

# Concatenate
dataframe = concat_data(load_data, weather_data)

# Renaming columns for easier interpreting:
dataframe = dataframe.rename(columns={"Total":"Load","Middeltemperatur i 2m høyde (TM)": "Temperature"})


# In[ ]:


dataframe.head()


# In[ ]:


def show_plots(data, time_start, time_end=None):
    # Ploting time-series data with different time ranges
    fig, ax = plt.subplots(figsize=(7,4.5))
    ax2 = ax.twinx()
    data['Load'].loc[time_start:time_end].plot(c='seagreen', label='Load', ax=ax)
    if time_end is None:
        data['Temperature'].loc[time_start].plot(c='darkorange', label='Temperature', ax=ax2)
    else:
        data['Temperature'].loc[time_start:time_end].plot(c='darkorange', label='Temperature', ax=ax2)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.set_ylabel('Load', fontsize=12, fontweight='bold', color='seagreen')
    ax2.set_ylabel('Temperature', fontsize=12, fontweight='bold', color='darkorange')
    fig.tight_layout()
    plt.show()
    return


# In[ ]:


# Time-series for 2018
show_plots(dataframe, '2017')


# In[ ]:


# Time-series for June to August 2018
show_plots(dataframe, '2018-06', '2018-08')


# In[ ]:


show_plots(dataframe, '2018-06-01', '2018-06-07')


# In[ ]:


# Hard-coding holiday dates
holiday_dates = {
    # 2014 year
    '2014-Jan-1st': ('2014-01-01', None),  # single day
    '2014-Easter': ('2014-04-14', '2014-04-21'),  # date range
    '2014-May-1st': ('2014-05-01', None),
    '2014-Pentecost': ('2014-06-07', '2014-06-10'),
    '2014-Xmas': ('2014-12-21', '2014-12-31'),
    # 2015 year
    '2015-Jan-1st': ('2015-01-01', None),
    '2015-Easter': ('2015-03-30', '2015-04-06'),
    '2015-May-1st': ('2015-05-01', None),  # Friday
    '2015-Ascension': ('2015-05-14', None),
    '2015-Pentecost': ('2014-05-24', '2014-05-25'),
    '2015-Xmas': ('2015-12-23', '2015-12-31'),
    # 2016 year
    '2016-Jan-1st': ('2016-01-01', None),
    '2016-Easter': ('2016-03-21', '2016-03-28'),
    '2016-May-1st': ('2015-05-01', None),  # Sunday
    '2016-Ascension': ('2016-05-05', None),
    '2016-Pentecost': ('2016-05-16', '2016-05-17'),
    '2016-Xmas': ('2016-12-26', '2016-12-31'),
    # 2017 year
    '2017-Jan-1st': ('2017-01-01', None),
    '2017-Easter': ('2017-04-10', '2017-04-17'),
    '2017-May-1st': ('2017-05-01', None),  # Monday
    '2017-May-17th': ('2017-05-17', None),  # Wednesday
    '2017-Ascension': ('2017-05-25', None),
    '2017-Pentecost': ('2017-06-05', None),
    '2017-Xmas': ('2017-12-25', '2017-12-31'),
    # 2018 year
    '2018-Jan-1st': ('2018-01-01', None),
    '2018-Easter': ('2018-03-26', '2018-04-02'),
    '2018-May-1st': ('2018-05-01', None),  # Tuesday
    '2018-Ascension': ('2017-05-10', None),  # Thursday
    '2018-May-17th': ('2017-05-17', None),
    '2018-Pentecost': ('2018-05-21', None),
    '2018-Xmas': ('2018-12-24', '2018-12-31')}


# In[ ]:


def engineer_features(dataframe, holiday_dates, columns, time_lags=24, 
                      drop_nan_rows=True):
    """Engineering features
    
    Parameters
    ----------
    dataframe: pandas dataframe
        original dataframe with time-series data
    holiday_dates: dictionary
        dictionary with tuples specifying local holiday dates or date-ranges
    columns: list
        list of column names from the dataframe which are used for the 
        features engineering (i.e. time-lags)
    time_lags: int
        number of time lags for use with feature engineering
    drop_nan_rows: bool
        True/False indicator to drop rows with NaN values
    
    Returns
    -------
    dataframe: pandas dataframe 
        dataframe augmented with additional features 
    """
    
    # Make a copy of the original dataframe
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

    # Weekend as a binary indicator
    data['weekend'] = np.asarray([0 if ind <= 4 else 1 for ind in weekday_ind])

    # Month indicators with cyclical transform
    month_ind = data.index.month
    data['mnth_sin'] = np.sin((month_ind-1)*(2.*np.pi/12))
    data['mnth_cos'] = np.cos((month_ind-1)*(2.*np.pi/12))
    
    # Holidays as a binary indicator
    data['holidays'] = 0
    for holiday, date in holiday_dates.items():
        if date[1] is None:
            # Single day
            data.loc[date[0], 'holidays'] = 1
        else:
            # Date range
            data.loc[date[0]:date[1], 'holidays'] = 1

    if drop_nan_rows:
        # Drop rows with NaN values
        data.dropna(inplace=True)
        
    return data


# In[ ]:


data_features = engineer_features(dataframe, holiday_dates, columns=['Load', 'Temperature'])
data_features.head()


# In[ ]:


print(data_features.columns)


# In[ ]:




