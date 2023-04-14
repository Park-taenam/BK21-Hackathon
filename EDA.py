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

def add_yield_diff(plant_info_df, plant_power_dict):
    for key, value in plant_power_dict.items():
        plant_power_df = plant_power_dict[key]

        plant_power_df['volume'] = list(plant_info_df.loc[plant_info_df['구분자'] == plant_power_df['Plant'].unique()[0], '용량'])[0]

        for inverter in plant_power_df['Inverter'].unique():
            inverter_idx = (plant_power_df['Inverter'] == inverter)
            plant_power_df['yield_diff_1'] = (plant_power_df.loc[inverter_idx, 'Total Yield(kWh)']).diff()
            plant_power_df['yield_diff_2'] = (plant_power_df.loc[inverter_idx, 'yield_diff_1']).diff()    

    return plant_power_dict
# %%
if __name__ == "__main__":
    ## Road Data
    plant_info_df, weather_dict, plant_power_dict = road_data()

    ## Yield Diff
    plant_power_dict = add_yield_diff(plant_info_df, plant_power_dict)

    ## Visualization
    # Plant Power - Total Yield
    for plant in list(plant_power_dict.keys()):
        try:
            draw_totalYield(plant_power_dict, plant)
        except Exception as e: 
            print("Error : {}".format(plant))
            print(e)
    # %%
    for key, value in plant_power_dict.items():
        value['Plant'] = key


    import pickle

    with open('test.pkl', 'wb') as fp:
        pickle.dump(plant_power_dict, fp)
        print('dictionary saved successfully to file')


    # Read dictionary pkl file
    with open('test.pkl', 'rb') as fp:
        test = pickle.load(fp)
        print('Person dictionary')

    test_df = plant_power_dict['ds'].loc[plant_power_dict['ds']['Inverter']=='KACO01', :].reset_index().iloc[:, 1:]