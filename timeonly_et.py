import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import pickle
import matplotlib.colors as mcolors
import re
from sklearn.ensemble import ExtraTreesRegressor
from et_utils import *

#---------- plant 별 : 5일치 시간단위 예측 ----------
#------ plot 함수 ------
def hourly_comparison_plot(data, plant) : 
    plt.plot(data['time'], data['yield_diff_1'], label = 'real', color = 'salmon')
    plt.plot(data['time'], data['pred_yield_diff_1'], label = 'pred', color = 'dodgerblue')
    plt.xlabel('time')
    plt.ylabel('hourly energy yield')
    plt.legend(['real', 'pred'], loc = 'upper right')
    plt.title(f'{plant} timeonly_hourly_prediction')
    plt.savefig('./{}_timeonly.png'.format(plant))
    plt.show()

#---------- plant 별 : 5일치 시간단위 예측 ----------
def hourly_ts_extratree(data):
    # 5일 치 시간단위로 예측한다 
    data = data[['hour_sin','hour_cos','day_sin','day_cos', 'time','yield_diff_1']]
    train = data[:-120]
    train_x = train[['hour_sin','hour_cos','day_sin','day_cos']]
    train_y = train['yield_diff_1']
    test = data[-120:]
    test_x = test[['hour_sin','hour_cos','day_sin','day_cos']]
    test_y = test['yield_diff_1']
    
    et = ExtraTreesRegressor(n_estimators = 100, max_depth = 10, random_state= 42)
    et.fit(train_x, train_y)
    test['pred_yield_diff_1'] = et.predict(test_x)

    return train, test

def predict_by_plants_inverter(data):
    data['plant_inverter'] = data['Plant'] + '-' + data['Inverter']
    # 평가지표 기록할 데이터프레임 생성 
    hourly_plant = pd.DataFrame(columns = ['plant_inverter','SMAPE' , 'RMSE' ,'R2'])

    for i in data['plant_inverter'].unique():
        data_inverter = data[data['plant_inverter'] == i ]
        train_plant , test_plant = hourly_ts_extratree(data_inverter)
        hourly_comparison_plot(test_plant, i)
        error_smape , error_rmse, error_r2 = evaluation(test_plant['yield_diff_1'], test_plant['pred_yield_diff_1'])
        hourly_plant.loc[i,'plant_inverter'] = i
        hourly_plant.loc[i , 'SMAPE'] = error_smape
        hourly_plant.loc[i , 'RMSE'] = error_rmse
        hourly_plant.loc[i , 'R2'] = error_r2
        
        
    return hourly_plant

# %%
if __name__=="__main__":
    #----- 데이터 불러오기 -----
    with open('./final_data.pkl', 'rb') as fp:
        data = pickle.load(fp)
    df = pd.DataFrame(columns = data['nk'].columns)
    for i in data.keys():
        df = pd.concat([df, data[i]])

    # ----- 데이터들을 하나의 데이터프레임으로 묶어주기 ------
    df = pd.DataFrame(columns = data['nk'].columns)
    for i in data.keys():
        df = pd.concat([df, data[i]])
    df['time'] = [str(i) for i in df['time']]
    # 'Date' 는 시간 외에 날짜 데이터만 포함하게 만듬
    df['date'] = [ i[:10] for i in df['time']]
    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour
    # 필요한 변수들만 추출 
    variables = ['Plant','Inverter','time','date','hour','Total Yield(kWh)','volume', 'yield_diff_1', 'yield_diff_2', 'percent','기온(°C)', '강수량(mm)', '풍속(m/s)',
        '풍향(16방위)', '습도(%)', '현지기압(hPa)', '일조(hr)', '일사(MJ/m2)', '적설(cm)',
        '전운량(10분위)', '지면온도(°C)']

    df = df[variables]
    create_features(df)
        
    result = predict_by_plants_inverter(df)
    result.to_csv('./time_et_result.csv')