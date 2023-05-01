# %%
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pickle

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
    weather_list = os.listdir(path)
    weather_dfs = {'영암': [], '무안': [], '해남': []}
    for weather in weather_list:
        # 아래 줄은 맥북일 때 실행
        weather = normalize('NFC', weather)
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
        if files == [] or plantName == '덕송태양광발전소':
            files = glob.glob(filepath + "/*.xls")
            df_list = [pd.read_excel(file) for file in files]
        else:
            df_list = [pd.read_csv(file, encoding='utf-8') for file in files]
        df = pd.concat(df_list, ignore_index=True)
        plantName = normalize('NFC', plantName)
        plant_dfs[plantName] = df
    return plant_dfs

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
    plantName = normalize('NFC', plantName)
    plt.title(plantName)
    plt.show()
    return

def save_weather_pickle():
    info = get_plant_info()
    weather_dfs = get_weather_dfs()
    path = os.getcwd() + '/geonhui_weather_data/'
    for key in weather_dfs.keys():
        weather_dfs[key].to_pickle(path + key + '.pkl')
    return

def get_plant_dict():
    # 각 발전소의 발전량 데이터프레임 pickle을 dictionary 형태로 저장.
    # 이때 get_plant_info() 데이터프레임의 '구분자'를 get_plant_dfs() 데이터프레임의 'Plant' 열에 적용.
    info = get_plant_info()
    plant_dfs = get_plant_dfs()
    new_plants = {}
    for key in plant_dfs.keys():
        if key[:2] == '동정':
            df_name = 'dj'
        else:
            for i in range(info.shape[0]):
                x = info.iloc[i, 1].replace(' ', '')
                if x[:4] == key[:4]:
                    df_name = info.iloc[i, 0]
        plant_dfs[key]['Plant'] = df_name
        new_plants[df_name] = plant_dfs[key]
    for key, df in new_plants.items():
        df = df.sort_values(['Inverter', 'Date'], ascending=True)
        df = df.reset_index(drop=True)
        time = pd.to_datetime(df['Date'])
        df.drop('Date', axis=1, inplace=True)
        df.insert(0, 'time', time)
        df['Plant'] = key
        new_plants[key] = df
    
    return new_plants

def plot_3regions_weather(drop_col=None):
    # 영암, 무안, 해남 3개 지역의 기상 데이터 시각화. drop_col 인자에 제외하고 싶은 column 추가 가능
    weather_dfs = get_weather_dfs()
    drop_col = drop_col
    for key, value in weather_dfs.items():
        temp_df = value
        if drop_col:
            temp_df = temp_df.drop(drop_col, axis=1)
        plt.figure(figsize=(15, 8))
        plt.plot(temp_df, label=temp_df.columns)
        plt.title(key)
        plt.legend()
        plt.show()
    return

def get_empty_index_dict():
    # 결측치가 존재하는 발전소에 대한 발전량 결측치 인덱스(hour 단위)를 딕셔너리 형태로 반환.
    plant_dict = get_plant_dict()
    empty_index_dict = {}
    for key, value in plant_dict.items():
        empty_index_dict[key] = pd.DataFrame()
        inverters = sorted(list(set(value['Inverter'])))
        for inv in inverters:
            plant = value.loc[value['Inverter'] == inv].copy()
            plant_timerange = plant['time']
            real_timerange = pd.date_range(start=plant_timerange.iloc[0],
                                        end=plant_timerange.iloc[-1],
                                        freq='H')
            s1 = set(list(plant_timerange))
            s2 = set(list(real_timerange))
            temp = list(set.symmetric_difference(s1, s2))
            if temp == []:
                if key in empty_index_dict:
                    del empty_index_dict[key]
                continue
            diff = pd.Series(temp)
            diff = diff.sort_values().reset_index(drop=True)
            # for x in diff:
            #     print(inv, x)
            diff = pd.DataFrame({inv: diff})
            if not diff.empty:
                empty_index_dict[key] = pd.concat([empty_index_dict[key], diff], axis=1)
    return empty_index_dict

def plot_missing_value(empty_index_dict):
    # 결측치 있는 발전소들 어느 시간대에 결측치 생겼는지 시각화
    idx = pd.date_range(start='2020-10-25 00:00:00',
                        end='2021-06-30 23:00:00',
                        freq='H')
    view_empty = pd.DataFrame(index=idx, columns=empty_index_dict.keys())
    col = []
    n = 0
    for key, value in empty_index_dict.items():
        temp = pd.DataFrame()
        for inv in value:
            temp = pd.concat([temp, value[inv]])
        temp = temp.dropna()[0].unique()
        view_empty.loc[temp, key] = n
        n += 1

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_yticks(list(range(view_empty.shape[1])))
    ax.set_yticklabels(view_empty.columns, fontsize=15)
    plt.plot(view_empty.iloc[:-1, :], marker='o' ,markersize=3)
    plt.grid(True, axis='y', alpha=0.5, linestyle='--')
    plt.title('결측치가 있는 발전소에 대해 시간별로 결측치의 유무를 표시한 그래프')
    plt.show()
    return

def get_filled_plant_dict():
    # get_plant_dict() 반환값의 각 데이터프레임 중 Total Yield(kWh) 결측치를 등차수열로 채움
    filled_plant_dict = get_plant_dict()
    empty_index_dict = get_empty_index_dict()
    for key in empty_index_dict.keys():
        # print(key)
        df = filled_plant_dict[key]
        inverters = sorted(df['Inverter'].unique())
        filled_df = pd.DataFrame()
        for inv in inverters:
            sub_df = df.loc[df['Inverter'] == inv, :]
            expand_range = pd.Series(pd.date_range(start=sub_df.iloc[0, 0],
                                                   end=sub_df.iloc[-1, 0],
                                                   freq='H'))
            expand_range.name = 'time'
            cont_df = pd.merge(expand_range,
                               sub_df,
                               how='left',
                               left_on='time',
                               right_on='time')
            idxs = cont_df.loc[cont_df['Plant'].isna(), :].index
            # 우선 각 결측치 바로 이전 값으로 채우기
            cont_df = cont_df.fillna(method='pad')
            # 결측치 위아래 인덱스 뽑기
            i = 0
            start_end_list = []
            while i < len(idxs):
                start = idxs[i] - 1
                k = 1
                if i != len(idxs) - 1:
                    while idxs[i+k] == idxs[i] + k:
                        k += 1
                        if i+k >= len(idxs):
                            break
                end = idxs[i] + k
                start_end_list.append([start, end])
                i += k
            # print(start_end_list)
            # 'Total Yield(kWh)' 열 결측치 위아래 값을 양끝값으로 하는 등차수열 채우기
            for start, end in start_end_list:
                length = end - start + 1
                fill = np.linspace(cont_df.iloc[start, 3],
                                   cont_df.iloc[end, 3],
                                   length)
                fill = fill[1:-1]
                for i in range(len(fill)):
                    cont_df.iloc[start+i+1, 3] = fill[i]
            filled_df = pd.concat([filled_df, cont_df], axis=0)
        filled_df.reset_index(drop=True, inplace=True)
        filled_plant_dict[key] = filled_df
    return filled_plant_dict


def get_net_weather():
    # 기상 데이터는 누락된 시간 없어서 value 채우는 과정 바로 진행
    weather_dfs = get_weather_dfs()
    for key, value in weather_dfs.items():
        value['region'] = key
        value['time'] = value.index
    sub_weather = [we for we in weather_dfs.values()]
    net_weather = pd.concat(sub_weather, ignore_index=True)
    net_weather['month'] = net_weather['time'].dt.month
    net_weather['day'] = net_weather['time'].dt.day
    net_weather['hour'] = net_weather['time'].dt.hour
    return net_weather


def plot_6_months_3_region(net_weather, col):
    # 6개월, 3개 지역(영암, 무안, 해남)에 대한 기상 데이터 시각화
    net_weather = net_weather
    fig, axis = plt.subplots(2, 3, sharey=True, figsize=(20, 10))
    for i in range(6):
        df = net_weather.loc[net_weather['month'] == i+1]
        # 해당 달의 col 원소들이 전부 NaN이면 건너뛰기
        if df[col].isna().all():
            continue
        # ax = axis[int(i/2), int(i%2)]
        ax = axis[int(i/3), int(i%3)]
        sns.lineplot(x='hour', y=col, data=df, hue='region', ax=ax)
        ax.set_title(f'{i+1}월 {col} 분포')
        ax.set_xlim(0,23)
        ax.set_xticks(range(0,24))
    plt.plot()
    return

def plot_6_3_NaN(net_weather, col):
    # 6개월, 3개 지역(영암, 무안, 해남)에 대한 기상 데이터 NaN 값 시각화
    net_weather = net_weather
    fig, axis = plt.subplots(2, 3, sharey=True, figsize=(20, 10))
    for i in range(6):
        df = net_weather.loc[(net_weather['month'] == i+1) &
                             (net_weather[col].isna())]

        ax = axis[int(i/3), int(i%3)]
        sns.histplot(x='hour', data=df, hue='region', multiple="dodge", ax=ax, bins=24)
        ax.set_title(f'{i+1}월 {col} NaN 개수')
        ax.set_xlim(0,23)
        ax.set_xticks(range(0,24))
    plt.plot()
    return


def plot_nan_3regions(net_weather):
    # 결측치 있는 기상 데이터 어느 시간대에 결측치 생겼는지 시각화
    net_weather = net_weather
    for region in net_weather['region'].unique():
        wth = net_weather.loc[net_weather['region']==region, :]
        wth = wth.iloc[:, :13]
        idx = pd.date_range(start='2021-01-01 00:00:00',
                            end='2021-06-30 23:00:00',
                            freq='H')
        view_nan_wth = pd.DataFrame(index=idx, columns=wth.iloc[:, :11].columns)
        n = 0
        for col in wth.columns:
            temp = wth.loc[wth[col].isna(), 'time']
            if temp.any():
                view_nan_wth.loc[temp, col] = n
                n += 1

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1)
        view_nan_wth = view_nan_wth.dropna(axis=1, how='all')
        ax.set_yticks(list(range(view_nan_wth.shape[1])))
        ax.set_yticklabels(view_nan_wth.columns, fontsize=15)
        plt.plot(view_nan_wth, marker='o' ,markersize=2)
        plt.grid(True, axis='y', alpha=0.5, linestyle='--')
        plt.title(f'{region} 시간별 기상 데이터 결측치')
        plt.show()
    return


def fill_nan_weather(col_idx, start=6, end=9):
    # 기온, 현지기압, 지면온도 채우는 등차수열 값 반환
    net_weather = get_net_weather()
    day_weather = pd.DataFrame(net_weather.iloc[816:840, :])
    day_weather = day_weather.reset_index(drop=True)
    
    col_name = net_weather.columns[col_idx]
    time_range = list(range(start-1, end+2))
    sub_weather_df = net_weather.loc[(net_weather['month'] == 2) & 
                        (net_weather['region'] == '영암') & 
                        (net_weather['day'] >= 1) & 
                        (net_weather['day'] <= 7) &
                        (net_weather['hour'].isin(time_range))].copy()
    mean = sub_weather_df.loc[(sub_weather_df['hour'].isin(time_range[1:-1]))].copy()
    mean = mean[col_name].groupby(mean['hour']).mean()
    arith_weather_df = pd.DataFrame(sub_weather_df[col_name])
    arith_weather_df.reset_index(drop=True, inplace=True)

    arith = np.linspace(day_weather.iloc[start-1, col_idx], 
                        day_weather.iloc[end+1, col_idx], 
                        len(time_range))[1:-1]
    df = net_weather.loc[(net_weather['month'] == 2) & 
                        (net_weather['region'] == '영암') & 
                        (net_weather['day'] >= 1) & 
                        (net_weather['day'] <= 7)]
    arith_plus_mean = arith * 0.7 + mean * 0.3
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='hour', y=col_name, data=df, label='2월 1일~7일 시간별 평균값')
    plt.plot(day_weather.iloc[:, col_idx], label='2월 4일 하루 측정치')
    plt.plot(arith * 1.0 + mean * 0, label='선형보간법')
    # plt.plot(mean)
    plt.title(f'시간별 {col_name} 분포')
    plt.legend()
    plt.show()
    
    return pd.Series(arith), arith_plus_mean, mean


def get_filled_weather_dict(net_weather):
    # 결측치가 채워진 기상 데이터 반환
    filled_weather = net_weather
    cols = net_weather.columns
    for n in [0, 5, 10]:
        # 기온, 현지기압, 지면온도는 결측치 양옆 값을 양끝으로 하는 등차수열 값으로 채움
        arith, _, _ = fill_nan_weather(n)
        filled_weather.loc[filled_weather.loc[:, cols[n]].isna(), cols[n]] = arith.values
    for n in [1, 6, 7, 8, 9]:
        # 강수량, 일조, 일사, 적설, 전운량은 결측치를 0.0으로 채움
        filled_weather.loc[filled_weather.loc[:, cols[n]].isna(), cols[n]] = 0.0
    filled_weather_dict = {}
    for region in list(set(filled_weather['region'])):
        sub_df = filled_weather.loc[filled_weather['region']==region, :]
        filled_weather_dict[region] = sub_df
    return filled_weather_dict


def get_plant_with_weather():
    # 발전량 데이터와 기상 데이터를 합친 뒤 1차차분/2차차분 및 하루 생산량 비율을 계산한 데이터프레임들을 딕셔너리 형태로 반환
    net_weather = get_net_weather()
    info = get_plant_info()
    filled_weather_dict = get_filled_weather_dict(net_weather)
    info['region'] = 0
    regions = list(filled_weather_dict.keys())
    for i in range(info.shape[0]):
        for region in regions:
            if region in info.iloc[i, 2]:
                info.iloc[i, -1] = region
        # 나주시: 기상자료개방포털 조사 결과 나주 기상 데이터는 없었고 전남 지역 중 나주와 가장 가까운 지역이 영암이므로 나주시 기상 데이터는 영암 데이터로 대체.
        if info.iloc[i, -1] == 0:
            info.iloc[i, -1] = '영암'
    # 'dj' 발전소가 info 데이터프레임에 없어서 해당 데이터의 '무안 서호리'를 참고하여 아래와 같이 추가.
    dj_temp = pd.DataFrame({'구분자': ['dj'], 'region': ['무안']})
    info = pd.concat([info, dj_temp], axis=0, ignore_index=True)
    plant_with_weather = get_filled_plant_dict()
    plant_index = list(plant_with_weather.keys())
    for idx in plant_index:
        for i in range(info.shape[0]):
            if idx == info.iloc[i, 0]:
                weather_df = filled_weather_dict[info.iloc[i, -1]]
                #w = weather_df['time']
                #w = list(map(lambda x: x[0], w.astype('str').str.split(':')))
                #w = pd.Series(w)
                #weather_df.index = w
                s = pd.merge(plant_with_weather[idx], weather_df, left_on='time', right_on='time', how='inner')
                s = s.reset_index(drop=True)
                plant_with_weather[idx] = s
    for name, df in plant_with_weather.items():
        df = df.sort_values(['Inverter', 'time'], ascending = True)
        df = df.reset_index(drop=True)
        if name == "dj":
            df['volume'] = 98.770
        else:
            df['volume'] = float(info.loc[info['구분자']==name, '용량'])
        inverters = sorted(df['Inverter'].unique())
        for inv in inverters:
            plant_inverter_df = df.loc[df['Inverter']==inv, :]
            plant_inverter_df = plant_inverter_df.sort_values(by='time', ascending=True)
            
            # diff 1
            yield_diff_1 = plant_inverter_df['Total Yield(kWh)'].diff().fillna(0)
            df.loc[yield_diff_1.index, 'yield_diff_1'] = yield_diff_1
            
            yield_diff_2 = df['yield_diff_1'].diff().fillna(0)
            df.loc[yield_diff_2.index, 'yield_diff_2'] = yield_diff_2
        
        df['percent'] = df['yield_diff_1'] / df['volume']
        
        plant_with_weather[name] = df
        
    return plant_with_weather


def draw_totalYield2(plants, plantName):
    # 발전량 데이터프레임을 저장한 딕셔너리와 발전소 이름을 인자값으로 넣으면 해당 발전소의 Total Yield(kWh) 값을 Inverter별, 그리고 월별로 plot함.
    # get_plant_with_weather() 함수를 통해 나온 최종 dictionary를 plants 인자로 넣으면 됨. 
    subdata = plants[plantName].copy()[['Inverter', 'time', 'Total Yield(kWh)']]
    subdata['Md'] = subdata['time'].dt.month
    subdata = subdata.sort_values(by='time')
    subdata = subdata.reset_index()
    
    hues = subdata[['Inverter', 'Md']].apply(tuple, axis=1)
    palette = sns.color_palette("Paired", n_colors=len(set(hues)))
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=subdata, x=subdata.index, y='Total Yield(kWh)', hue=hues, palette=palette)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plantName = normalize('NFC', plantName)
    plt.title(plantName)
    plt.show()
    return

# %%
if __name__=='__main__':
    plant_with_weather = get_plant_with_weather()

    draw_totalYield2(plant_with_weather, 'hb')

    """
    # 데이터 저장하기
    with open('final_data_geonhui.pkl', 'wb') as fp:
            pickle.dump(plant_with_weather, fp)
            print('dictionary saved successfully to file')
    """

    """
    # 데이터 불러오기
    with open('final_data_geonhui.pkl', 'rb') as fp:
        data = pickle.load(fp)
    """