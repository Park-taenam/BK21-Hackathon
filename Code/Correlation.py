#%% 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import pickle
import matplotlib.colors as mcolors
import re
from matplotlib import rc

def draw_correlation_plot(data):
    corr_variables = ['yield_diff_1','기온(°C)', '강수량(mm)', '풍속(m/s)',
    '풍향(16방위)', '습도(%)', '현지기압(hPa)', '일조(hr)', '일사(MJ/m2)', '적설(cm)',
    '전운량(10분위)', '지면온도(°C)']
    data_corr = data[corr_variables]
    df_corr = data_corr.apply(pd.to_numeric)
    correlation_mat = df_corr.corr()

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    # More details at https://seaborn.pydata.org/generated/seaborn.heatmap.html
    sns.heatmap(
        correlation_mat,          # The data to plot    # Mask some cells
        cmap=cmap,     # What colors to plot the heatmap as
        annot=True,    # Should the values be plotted in the cells?
        vmax=.3,       # The maximum value of the legend. All higher vals will be same color
        vmin=-.3,      # The minimum value of the legend. All lower vals will be same color
        center=0,      # The center value of the legend. With divergent cmap, where white is
        square=True,   # Force cells to be square
        linewidths=.5, # Width of lines that divide cells
        cbar_kws={"shrink": .5}  # Extra kwargs for the legend; in this case, shrink by 50%
    )   

# %%
if __name__=="__main__":
    # ----- 데이터 불러오기 -----
    with open('./final_data.pkl', 'rb') as fp:
        data = pickle.load(fp)


    # ----- 데이터들을 하나의 데이터프레임으로 묶어주기 ------
    df = pd.DataFrame(columns = data['nk'].columns)
    for i in data.keys():
        df = pd.concat([df, data[i]])
    df['time'] = [str(i) for i in df['time']]
    df.rename(columns = {'time':'Date_Hour'}, inplace = True)
    # 'Date' 는 시간 외에 날짜 데이터만 포함하게 만듬
    #df['Date'] = [ i[:10] for i in df['Date_Hour']]
    

    #------- 헌굴 꺄잠 방지 
    rc('font', family='AppleGothic') 			
    plt.rcParams['axes.unicode_minus'] = False  # %%

    draw_correlation_plot(df)
