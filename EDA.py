# %%
import numpy as np
import pandas as pd

# %% 
## 발전소
plant = pd.read_csv('./Data/발전소 테이블.csv')
plant.columns = ['순번', '구분자', '발전소명', '주소', '위도', '경도', '위치좌표', '위치좌표', '용량']
plant.to_pickle('./Data/plant_info.pkl')
# %% 
## 기상데이터
# 전남 강진 영암 기상데이터 (강진만 있음)
weather_1_1 = pd.read_excel('./Data/기상데이터/202101_전남 강진 영암 기상 데이터.xls')
weather_1_2 = pd.read_excel('./Data/기상데이터/202102_전남 강진 영암 기상 데이터.xls')
weather_1_3 = pd.read_excel('./Data/기상데이터/202103_전남 강진 영암 기상 데이터.xls')
weather_1_4 = pd.read_excel('./Data/기상데이터/202104_전남 강진 영암 기상 데이터.xls')
weather_1_5 = pd.read_excel('./Data/기상데이터/202105_전남 강진 영암 기상 데이터.xls')
weather_1_6 = pd.read_excel('./Data/기상데이터/202106_전남 강진 영암 기상 데이터.xls')
weather_1 = pd.concat([weather_1_1, weather_1_2, weather_1_3, weather_1_4, weather_1_5, weather_1_6])
weather_1 = weather_1.reset_index()

# 전남 무안 목포 기상데이터 (목포만 있음)
weather_2_1 = pd.read_excel('./Data/기상데이터/202101_전남 무안 목포 기상 데이터.xls')
weather_2_2 = pd.read_excel('./Data/기상데이터/202102_전남 무안 목포 기상 데이터.xls')
weather_2_3 = pd.read_excel('./Data/기상데이터/202103_전남 무안 목포 기상 데이터.xls')
weather_2_4 = pd.read_excel('./Data/기상데이터/202104_전남 무안 목포 기상 데이터.xls')
weather_2_5 = pd.read_excel('./Data/기상데이터/202105_전남 무안 목포 기상 데이터.xls')
weather_2_6 = pd.read_excel('./Data/기상데이터/202106_전남 무안 목포 기상 데이터.xls')
weather_2 = pd.concat([weather_2_1, weather_2_2, weather_2_3, weather_2_4, weather_2_5, weather_2_6])
weather_2 = weather_2.reset_index()

# 전남 해남 기상데이터 (해남만 있음)
weather_3_1 = pd.read_excel('./Data/기상데이터/202101_전남 해남 기상 데이터.xls')
weather_3_2 = pd.read_excel('./Data/기상데이터/202102_전남 해남 기상 데이터.xls')
weather_3_3 = pd.read_excel('./Data/기상데이터/202103_전남 해남 기상 데이터.xls')
weather_3_4 = pd.read_excel('./Data/기상데이터/202104_전남 해남 기상 데이터.xls')
weather_3_5 = pd.read_excel('./Data/기상데이터/202105_전남 해남 기상 데이터.xls')
weather_3_6 = pd.read_excel('./Data/기상데이터/202106_전남 해남 기상 데이터.xls')
weather_3 = pd.concat([weather_3_1, weather_3_2, weather_3_3, weather_3_4, weather_3_5, weather_3_6])
weather_3 = weather_3.reset_index()

weather_1.to_pickle('./Data/gangjin_weather.pkl')
weather_2.to_pickle('./Data/mokpo_weather.pkl')
weather_3.to_pickle('./Data/haenam_weather.pkl')
# %%