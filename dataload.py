# %%
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob

import warnings
warnings.filterwarnings('ignore')

# %%
# 맥북인 경우만 아래 블럭 실행
from unicodedata import normalize
from matplotlib import rc
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False


# %%
def get_weather_dfs():
    # 2020.1~ 2020.6의 날씨 데이터를 3개 지역에 대해 저장한 딕셔너리 형태
    # 반환값: {'강진': 날씨df, '무안': 날씨df, '해남': 날씨df}
    path = os.getcwd() + '/에이아이씨티/기상데이터'
    weather_dfs = {}
    weather_list = os.listdir(path)
    weather_dfs = {'강진': [], '무안': [], '해남': []}
    for weather in weather_list:
        # 아래 줄은 맥북일 때 실행
        # weather = normalize('NFC', weather)
        for key in weather_dfs.keys():
            if key in weather:
                weather_dfs[key].append(os.path.join(path, weather))
    for key in weather_dfs.keys():
        files = sorted(weather_dfs[key])
        df_list = [pd.read_excel(file) for file in files]
        df = pd.concat(df_list, ignore_index=True)
        df = df.drop(['index', '지점', '지점명', '지역'], axis=1, errors='ignore')
        df = df.set_index('일시')
        weather_dfs[key] = df
    return weather_dfs


def get_plant_info():
    # 발전소 이름, 위치 등을 저장한 데이터프레임 형태
    # 반환값: 전처리한 '발전소 테이블.xls' df.
    path = os.getcwd() + '/에이아이씨티'
    for filename in os.listdir(path):
        if filename.endswith('xls'):
            filepath = os.path.join(path, filename)
    df = pd.read_excel(filepath)
    df = df.drop(0, axis=0)
    df = df.drop('순번', axis=1)
    df = df.reset_index(drop=True)
    return df


def get_plant_dfs():
    # 각 발전소의 발전량 데이터프레임을 저장한 딕셔너리 형태
    # 반환값: {'가나신화': 발전량df, ..., '형주태양광발전소': 발전량df}, 발전소 이름 리스트
    path = os.getcwd() + '/에이아이씨티/발전량 변환'
    plant_dfs = {}
    plantName_list = os.listdir(path)
    plantName_list.remove('.DS_Store')
    for plantName in plantName_list:
        filepath = os.path.join(path, plantName)
        files = glob.glob(filepath + "/*.csv")
        if files == []:
            files = glob.glob(filepath + "/*.xls")
            df_list = [pd.read_excel(file) for file in files]
        else:
            df_list = [pd.read_csv(file, encoding='utf-8') for file in files]
        df = pd.concat(df_list, ignore_index=True)
        plant_dfs[plantName] = df
    return plant_dfs, plantName_list


def draw_totalYield(plant_dfs, plantName):
    # 발전량 데이터프레임을 저장한 딕셔너리와 발전소 이름을 인자값으로 넣으면 해당 발전소의 Total Yield(kWh) 값을 Inverter별, 그리고 월별로 plot함.
    subdata = plant_dfs[plantName].copy()[['Inverter', 'Date', 'Total Yield(kWh)']]
    subdata['Md'] = subdata['Date'].apply(lambda x: x[:-6])
    subdata = subdata.sort_values(by='Date')
    subdata = subdata.reset_index()
    
    hues = subdata[['Inverter', 'Md']].apply(tuple, axis=1)
    palette = sns.color_palette("Paired", n_colors=len(set(hues)))
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=subdata, x=subdata.index, y='Total Yield(kWh)', hue=hues, palette=palette)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    return


def save_plant_pickle():
    # 각 발전소의 발전량 데이터프레임을 pickle 형태로 저장.
    # 이때 get_plant_info() 데이터프레임 중 '구분자'를 이름으로 하여 저장.
    info = get_plant_info()
    plant_dfs = get_plant_dfs()[0]
    path = os.getcwd() + '/powerplant/'
    for key in plant_dfs.keys():
        # 주석 블럭: 맥북 용
        """
        key_2 = normalize('NFC', key)
        if key_2[:2] == '동정':
            df_name = 'dj'
        else:
            for i in range(info.shape[0]):
                x = info.iloc[i, 1].replace(' ', '')
                if x[:4] == key_2[:4]:
                    df_name = info.iloc[i, 0]
        plant_dfs[key].to_pickle(path + df_name + '.pkl')
        """
        if key[:2] == '동정':
            df_name = 'dj'
        else:
            for i in range(info.shape[0]):
                x = info.iloc[i, 1].replace(' ', '')
                if x[:4] == key[:4]:
                    df_name = info.iloc[i, 0]
        plant_dfs[key].to_pickle(path + df_name + '.pkl')
    return

# %%
weather_dfs = get_weather_dfs()
for key, value in weather_dfs.items():
    temp_df = value
    temp_df = temp_df.drop(['현지기압(hPa)', '풍향(16방위)', '습도(%)', '전운량(10분위)'], axis=1)
    plt.figure(figsize=(15, 8))
    plt.plot(temp_df, label=temp_df.columns)
    plt.title(key)
    plt.legend()
    plt.show()

# %%
plant_dfs, plantName_list = get_plant_dfs()
draw_totalYield(plant_dfs, list(plantName_list)[5])

