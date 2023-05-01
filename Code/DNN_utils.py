import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

## Preprocessing
def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))

def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

def create_features(df):
    # time ,day encoding
    df['hour_sin'] = sin_transformer(24).fit_transform(df['time'].dt.hour)
    df['day_sin'] = sin_transformer(365).fit_transform(df['time'].dt.dayofyear)

    df['hour_cos'] = cos_transformer(24).fit_transform(df['time'].dt.hour)
    df['day_cos'] = cos_transformer(365).fit_transform(df['time'].dt.dayofyear)
    
    return df

def split_data_time(df, num=120):
    inverter_list = df['Inverter'].unique()
    test_idx = [(df.loc[df['Inverter'] == inverter].iloc[-num:, :]).index for inverter in inverter_list]
    test_idx = [x for indices in test_idx for x in indices ]

    train_idx = [(df.loc[df['Inverter'] == inverter].iloc[:-num, :]).index for inverter in inverter_list]
    train_idx = [x for indices in train_idx for x in indices]

    train_df = df.loc[train_idx, :]
    test_df = df.loc[test_idx, :]

    return inverter_list, train_df, test_df

def feature_selection(df_train, df_test):
    feat = ['Inverter', '기온(°C)', '강수량(mm)', '풍속(m/s)',
          '습도(%)', '현지기압(hPa)', '일조(hr)', '일사(MJ/m2)', '적설(cm)', 
          '전운량(10분위)', '지면온도(°C)', 'hour_sin', 'day_sin', 'hour_cos', 'day_cos']
    
    X_train, X_test = df_train[feat], df_test[feat]
    y_train, y_test = df_train['yield_diff_1'], df_test['yield_diff_1']

    return X_train, y_train, X_test, y_test

def one_hot_encoding(X_train, X_test):
    ## Inverter OnehotEncoding
    ohe = OneHotEncoder(sparse=False)
    ohe.fit(X_train.loc[:,['Inverter']])

    train_enc = ohe.transform(X_train.loc[:,['Inverter']])
    X_train[ohe.categories_[0]] = train_enc
    X_train = X_train.iloc[:, 1:]

    test_enc = ohe.transform(X_test.loc[:,['Inverter']])
    X_test[ohe.categories_[0]] = test_enc
    X_test = X_test.iloc[:, 1:]

    return X_train, X_test

def min_max_scaling(X_train, X_test):
    ## MinMaxScaling
    feat_list = list(X_train.columns)
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(X_train)

    X_train[feat_list] = min_max_scaler.transform(X_train)
    X_test[feat_list] = min_max_scaler.transform(X_test)

    return X_train, X_test

def evaluation(y_true, y_pred):
    smape = np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
    RMSE = mean_squared_error(y_true, y_pred)**0.5
    R2 = r2_score(y_true, y_pred)
    return smape, RMSE, R2

## plot
def Loss_plot(plant, trn_loss_list, val_loss_list):
    ## Loss visualization
    trn_loss_list = [x.cpu().detach().numpy() for x in trn_loss_list]
    val_loss_list = [x.cpu().detach().numpy() for x in val_loss_list]

    plt.figure(figsize=(16,9))
    x_range = range(len(trn_loss_list))
    plt.plot(x_range, trn_loss_list, label="trn")
    plt.plot(x_range, val_loss_list, label="val")
    plt.legend()
    plt.title("{} - Loss plot".format(plant))
    plt.xlabel("training steps")
    plt.ylabel("loss")
    plt.savefig('./Image/Loss/{}_Loss.png'.format(plant))
    plt.show()

    return

def hourly_comparison_plot(pred_df, plant_inverter): 
    plt.plot(pred_df.index, pred_df['y_test'], label = 'real', color = 'salmon')
    plt.plot(pred_df.index, pred_df['y_pred'], label = 'pred', color = 'dodgerblue')
    plt.xlabel('Index')
    plt.ylabel('hourly energy yield')
    plt.legend(['real', 'pred'], loc = 'upper right')
    plt.title(f'{plant_inverter} - hourly_prediction')
    plt.savefig('./Image/compare_real_pred/{}_pred_hourly.png'.format(plant_inverter))
    plt.show()
    
    return

def hourly_diff_comparison_plot(pred_df, plant) : 
    ## scatter plot
    y_min = min(pred_df['y_test'])
    y_max = max(pred_df['y_test'])

    pred_df = pred_df.sort_values(by='y_test')
    plt.axline((0, 0), slope=1, linewidth=1, color='r')

    plt.scatter(pred_df['y_test'], pred_df['y_pred'], color='darkmagenta', marker='o')
    plt.axis([y_min-2, y_max+2, y_min-2, y_max+2])
    plt.xlabel('Real value')
    plt.ylabel('Prediction')
    plt.title('{} - Test Prediction'.format(plant))
    plt.savefig('./Image/Scatter/{}_scatter.png'.format(plant))
    plt.show()

    return