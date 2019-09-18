#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# heatmap 热力图
def import_data():
    # path_file = 'accuracy_table2.csv'
    path_file = 'accuracy_table3.csv'
    data = pd.read_csv(path_file)
    print("the shape of {} is {}".format(path_file, data.shape))
    data = data.set_index('Accuracy')
    return data
photo = import_data()
#%%    
def plot_line(data):
    # x = np.arange(1,41,1)
    x = np.arange(1,31,1)
    sns.set_style("whitegrid")
    plt.rc('font',family='Times New Roman',weight='heavy')   
    plt.plot(x,data['BP+FE'].values,label='BP+FE',linewidth='1',linestyle='-',color='g',marker='o',markersize='5')
    plt.plot(x,data['CNN'].values,label='CNN',linewidth='1',linestyle='-',color='b',marker='v',markersize='5')
    plt.plot(x,data['CNN+FE'].values,label='CNN+FE',linewidth='1',linestyle='-',color='k',marker='*')
    plt.plot(x,data['CNN+FE+HD'].values,label='CNN+FE+HD',linewidth='1',linestyle='-',color='c',marker='s',markersize='4')
    plt.plot(x,data['Caps+FE+HD'].values,label='Caps+FE+HD',linewidth='1',linestyle='-',color='r',marker='^',markersize='5')
    plt.plot(x,data['HDACAPS*'].values,label='HDACAPS*',linewidth='1',linestyle='-',color='m',marker='d',markersize='4')
    
    # loss_table2.csv
    # plt.xlim([0,41])
    # plt.xticks([0,5,10,15,20,25,30,35,40])
    # loss_table3.csv
    plt.xlim([0,31])
    plt.xticks([0,5,10,15,20,25,30])

    plt.yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9'])
    plt.xlabel('Loops',fontsize=15,weight='heavy')
    plt.ylabel('Classification Accuracy',fontsize=15,weight='heavy')
    plt.legend(loc=0,ncol=1,fontsize=10)
    plt.tick_params(labelsize=15)
    # plt.savefig('Accuracy_table_2.png',dpi=800)
    plt.savefig('Accuracy_table_3.png',dpi=800)
    plt.show()
plot_line(photo)

