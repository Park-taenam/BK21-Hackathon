# %%
import os
import platform
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

op_sys = platform.system()
font_family = 'AppleGothic' if op_sys=='Darwin' else 'NanumGothic'
plt.rcParams['font.family'] = font_family

sns.set_context("talk")
sns.set_style("white")
sns.set_palette("Set1")
# %%
def road_data():
    plant_info_df = pd.read_pickle('./Data/plant_info.pkl')
    plant_list = plant_info_df['구분자']

    city_list = ['gangjin', 'haenam', 'mokpo']
    weather_dict = {}
    for city in city_list:
        weather_dict[city] = pd.read_pickle('./Data/weather/{}_weather.pkl'.format(city))

    plant_power_dict  = {}
    for plant in plant_list:
        plant_power_dict[plant] = pd.read_pickle('./Data/powerplant/{}.pkl'.format(plant))

    with open('./Data/filled_plant_with_weather.pkl', 'rb') as fp:
        plant_with_weather_dict = pickle.load(fp)

    return plant_info_df, weather_dict, plant_power_dict, plant_with_weather_dict
# %%
def draw_totalYield(plant_dfs, plantName):
    '''
    parameter : 발전량 데이터프레임을 저장한 딕셔너리, 발전소 이름
    output : 해당 발전소의 Total Yield(kWh) 값을 Inverter별, 그리고 월별로 plot
    '''
    subdata = plant_dfs[plantName].copy()[['Inverter', 'Date', 'Total Yield(kWh)']]
    subdata['Md'] = subdata['Date'].apply(lambda x: x[:-6])
    subdata = subdata.sort_values(by='Date')
    subdata = subdata.reset_index()
    
    hues = subdata[['Inverter', 'Md']].apply(tuple, axis=1)
    palette = sns.color_palette("Paired", n_colors=len(set(hues)))

    plt.figure(figsize=(12, 8))
    plt.title("{} - Total Yield".format(plantName))
    sns.lineplot(data=subdata,
                 x=subdata.index,
                 y='Total Yield(kWh)',
                 hue=hues,
                 palette=palette)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('./Image/TotalYield_{}'.format(plantName))
    plt.show()
    return

def add_yield_diff(plant_info_df, plant_with_weather_dict):
    for key in plant_with_weather_dict.keys():
        plant_df = plant_with_weather_dict[key]

        # create volume
        if key == "dj":
            plant_df['volume'] = 98.770
        else:
            plant_df['volume'] = list(plant_info_df.loc[plant_info_df['구분자'] == plant_df['Plant'].unique()[0], '용량'])[0]

        # create yield_diff 1, 2
        for inverter in plant_df['Inverter'].unique():
            inverter_idx = (plant_df['Inverter'] == inverter)
            plant_inverter_df = plant_df.loc[inverter_idx, :]
            plant_inverter_df = plant_inverter_df.sort_values(by='Date', ascending=True)
            
            # diff 1
            yield_diff_1 = plant_inverter_df['Total Yield(kWh)'].diff().fillna(0)
            plant_df.loc[yield_diff_1.index, 'yield_diff_1'] = yield_diff_1
            
            yield_diff_2 = plant_df['yield_diff_1'].diff().fillna(0)
            plant_df.loc[yield_diff_2.index, 'yield_diff_2'] = yield_diff_2
        
        plant_df['percent'] = plant_df['yield_diff_1']/plant_df['volume']
        
        plant_with_weather_dict[key] = plant_df  
    
    with open('./Data/final_data.pkl', 'wb') as fp:
        pickle.dump(plant_with_weather_dict, fp)
        print('dictionary saved successfully to file')

    return plant_with_weather_dict

# %%
if __name__ == "__main__":
    ## Road Data
    plant_info_df, weather_dict, plant_power_dict, plant_with_weather_dict = road_data()

    plant_name_list = list(plant_with_weather_dict.keys())

    ## Volume, Yield Diff
    plant_power_dict = add_yield_diff(plant_info_df, plant_with_weather_dict)

    ## Visualization
    # Plant Power - Total Yield
    for plant in list(plant_power_dict.keys()):
        try:
            draw_totalYield(plant_power_dict, plant)
        except Exception as e: 
            print("Error : {}".format(plant))
            print(e)