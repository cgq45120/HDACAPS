#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# heatmap 热力图
x = np.arange(1,37,1).reshape(6,6)/36
print(x)
#%%    
def plot_heat(data):
    sns.set_style("whitegrid")
    sns.heatmap(data)
    plt.rc('font',family='Times New Roman',weight='heavy')   
    plt.tick_params(labelsize=15)
    plt.savefig('heat_table_2.png',dpi=800)
    plt.show()
plot_heat(x)



#%%
