
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer

def create_features(df):
    df['hour_sin'] = sin_transformer(24).fit_transform(df['time'].dt.hour)
    df['day_sin'] = sin_transformer(365).fit_transform(df['time'].dt.dayofyear)
    df['hour_cos'] = cos_transformer(24).fit_transform(df['time'].dt.hour)
    df['day_cos'] = cos_transformer(365).fit_transform(df['time'].dt.dayofyear)
    
    return df

# -------- sin / cos화 코드 ------
def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

# ---------- 전처리 과정 ----------
def preprocess(data):
    data.reset_index(inplace = True)
    Train = data[1:-120]
    Test = data[-120:]
    x_num = ['기온(°C)','강수량(mm)', '풍속(m/s)', '습도(%)', 
         '일조(hr)', '일사(MJ/m2)', '전운량(10분위)', '지면온도(°C)']
    Train_x_num = Train[x_num]
    Test_x_num = Test[x_num]
    Train_y = Train[['yield_diff_1','yield_diff_2']]
    Test_y = Test[['yield_diff_1','yield_diff_2']]

    # fitting 은 train데이터에 대해서 진행
    mMscaler = MinMaxScaler()
    mMscaler.fit(Train_x_num)

    # 실제 scaling
    Train_x_num_scaled = pd.DataFrame(mMscaler.transform(Train_x_num), columns = x_num, index = Train_y.index)
    Test_x_num_scaled = pd.DataFrame(mMscaler.transform(Test_x_num), columns = x_num, index = Test_y.index)

    # 다시 데이터 병합 

    Train_preprocessed = pd.concat([Train[['yield_diff_1','yield_diff_2','time','hour_sin','hour_cos','day_sin','day_cos']], Train_x_num_scaled], axis = 1)
    Test_preprocessed = pd.concat([Test[['yield_diff_1','yield_diff_2','time','hour_sin','hour_cos','day_sin','day_cos']], Test_x_num_scaled], axis = 1)

    return Train_preprocessed, Test_preprocessed

# ----- evaluation metrics 생성 ------------ 
def evaluation(y_true, y_pred):
    smape = np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
    RMSE = mean_squared_error(y_true, y_pred)**0.5
    R_2 = r2_score(y_true, y_pred)
    return smape, RMSE, R_2