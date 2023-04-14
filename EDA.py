# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %%
def road_data():
    plant_df = pd.read_pickle('./Data/plant_info.pkl')
    plant_list = plant_df['구분자']

    city_list = ['gangjin', 'haenam', 'mokpo']
    weather_dict = {}
    for city in city_list:
        weather_dict[city] = pd.read_pickle('./Data/weather/{}_weather.pkl'.format(city))

    plant_power_dict  = {}
    for plant in plant_list:
        plant_power_dict[plant] = pd.read_pickle('./Data/powerplant/{}.pkl'.format(plant))

    return plant_df, weather_dict, plant_power_dict 
# %%
def draw_totalYield(plant_dfs, plantName):
    # 발전량 데이터프레임을 저장한 딕셔너리와 발전소 이름을 인자값으로 넣으면 해당 발전소의 Total Yield(kWh) 값을 Inverter별, 그리고 월별로 plot함.
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

def weather_plot(weather_dfs):
    for key, value in weather_dfs.items():
        temp_df = value
        temp_df = temp_df.drop(['현지기압(hPa)', '풍향(16방위)', '습도(%)', '전운량(10분위)'], axis=1)
        plt.figure(figsize=(15, 8))
        plt.plot(temp_df, label=temp_df.columns)
        plt.title(key)
        plt.legend()
        plt.show()

# %%
plant_df, weather_dict, plant_power_dict = road_data()

for plant in list(plant_power_dict.keys()):
    try:
        draw_totalYield(plant_power_dict, plant)
    except:
        print("Error : {}".format(plant))