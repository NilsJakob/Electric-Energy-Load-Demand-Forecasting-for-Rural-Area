{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "\n",
    "#function to read load data and weather data:\n",
    "def data_reader(file_name):\n",
    "    data = pd.read_excel(file_name, parse_dates=True, index_col='Time', usecols=range(2))\n",
    "    return data\n",
    "\n",
    "load_data = data_reader('C:/Users/nilsjj12/Documents/Present/Data/Index_Bjønntjønn_2014_2018.xlsx')\n",
    "\n",
    "def weather_reader(file_name):\n",
    "    weather = pd.read_excel(file_name, parse_dates=True, index_col='Time measured')\n",
    "    return weather\n",
    "\n",
    "weather_data = weather_reader('C:/Users/nilsjj12/Documents/Present/Data/bo_temp_2014_2018.xlsx')\n",
    "\n",
    "#function for concatenating load data and weather data for training:\n",
    "def data(file_name_load, file_name_weather):\n",
    "    train_data = pd.concat([file_name_load, file_name_weather], axis=1)\n",
    "    return train_data\n",
    "\n",
    "weather_data = weather_data.interpolate()\n",
    "Training = data(load_data, weather_data)\n",
    "#print(Training.head())\n",
    "\n",
    "#Renaming columns for easier interpreting:\n",
    "Training = Training.rename(columns={\"Total\":\"Load\",\"Middeltemperatur i 2m høyde (TM)\": \"Temperature\"})\n",
    "Training.describe()\n",
    "\n",
    "#Binary series to distuinguish working days from holidays by 1 and 0:\n",
    "s = pd.date_range('2014-01-01', '2019-01-01', freq='H').to_series()\n",
    "Training['weekday'] = s.dt.dayofweek\n",
    "#Training['weekday'] = Training['weekday'].astype(int)\n",
    "Training['working_days'] = Training['weekday'].replace({6: 1, 5: 1, 4: 1, 3: 0, 2: 0, 1: 0})\n",
    "\n",
    "#function to create sliding window based on time shifts:\n",
    "def time_shifts_func(name, data_hrs, time_shift, regr=False):\n",
    "    # name = 'DK1'\n",
    "    # time_shift = 24\n",
    "    if not regr:\n",
    "        data_hrs[name + '_t' + '+' + str(time_shift)] = data_hrs[name].shift(time_shift)\n",
    "        \n",
    "    else:\n",
    "        data_hrs['auto_' + name + '_t' + '+' + str(time_shift)] = (data_hrs[name].shift(time_shift)-data_hrs[name].shift(time_shift+1))\n",
    "    #print(data_hrs[name].shift(time_shift))\n",
    "    #data_hours['DK1_t+24'] = data_hours[name].shift(+24)\n",
    "    #data_hours['DK1_t+168'] = data_hours[name].shift(+168)\n",
    "    #data_hours['DK1_t-24'] = data_hours[name].shift(-24)\n",
    "    #return data_hrs\n",
    "time_shifts_func('Load', Training, 1)\n",
    "time_shifts_func('Load', Training, 2)   \n",
    "time_shifts_func('Load', Training, 24)\n",
    "time_shifts_func('Temperature', Training, 24)\n",
    "#time_shifts_func('Load - kWh', Training, 168)\n",
    "#time_shifts_func('Load - kWh', Training,  24, regr=True)\n",
    "#time_shifts_func('Load - kWh', Training,  1, regr=True)\n",
    "\n",
    "#Training=Training.dropna()\n",
    "\n",
    "#Training.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
