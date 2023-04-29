# %%
import pandas as pd
import matplotlib.pyplot as plt

def create_metric_mean(univar, multivar, dnn):
    metric_mean = pd.DataFrame()
    metric_mean['univar'] = univar.mean()
    metric_mean['multivar'] = multivar.mean()
    metric_mean['dnn'] = dnn.mean()

    metric_mean = metric_mean.T

    return metric_mean

def create_result_matrix(univar, multivar, dnn):
    a = pd.Series(univar.index).apply(lambda x: x[-6:])
    b = pd.Series(univar.index).apply(lambda x: x[:-7])
    univar.reset_index(drop=True, inplace=True)
    univar['inverter'] = a
    univar['plant'] = b

    a = pd.Series(multivar.index).apply(lambda x: x[-6:])
    b = pd.Series(multivar.index).apply(lambda x: x[:-7])
    multivar.reset_index(drop=True, inplace=True)
    multivar['inverter'] = a
    multivar['plant'] = b

    a = pd.Series(dnn.index).apply(lambda x: x[-6:])
    b = pd.Series(dnn.index).apply(lambda x: x[:-7])
    dnn.reset_index(drop=True, inplace=True)
    dnn['inverter'] = a
    dnn['plant'] = b

    univar_mean = univar.groupby(['plant']).mean()
    multivar_mean = multivar.groupby(['plant']).mean()
    dnn_mean = dnn.groupby(['plant']).mean()

    me_dict = {}
    me_dict['univar'] = univar_mean
    me_dict['multivar'] = multivar_mean
    me_dict['dnn'] = dnn_mean

    labels = ['ExTr (시간 데이터만 사용)', 'ExTr (시간+기상 데이터 사용)', 'DNN (시간+기상 데이터 사용)']
    for metric in univar_mean.columns:
        plt.figure(figsize=(8, 5))
        for i, value in enumerate(me_dict.values()):
            plt.plot(value[metric], label=labels[i])
            plt.legend()
        plt.title(f'{metric}')
        plt.show()
# %%
if __name__=="__main__":
    dnn = pd.read_csv('./result/DNN_result.csv', index_col=0)
    dnn.set_index('plant_inverter', inplace=True)
    dnn.index.name = None

    univar = pd.read_csv('./result/time_et_result.csv', index_col=0)
    univar.drop('plant_inverter', axis=1, inplace=True)

    multivar = pd.read_csv('./result/time_weather_et_result.csv', index_col=0)
    multivar.drop('plant_inverter', axis=1, inplace=True)

    metric_mean = create_metric_mean(dnn, univar, multivar)
    create_result_matrix(univar, multivar, dnn)
