import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd

data_name = 'HSLS'
imputer1 = 'Gain'
imputer2 = 'MICE'
imputer3 = 'KNN'
imputer_lst = [imputer1, imputer2, imputer3]

filename1 = './Results/Imputer_RMSE/Imputer_RMSE'+'_'+imputer1+'_'+data_name+'.csv'
filename2 = './Results/Imputer_RMSE/Imputer_RMSE'+'_'+imputer2+'_'+data_name+'.csv'
filename3 = './Results/Imputer_RMSE/Imputer_RMSE'+'_'+imputer3+'_'+data_name+'.csv'

df1 = pd.read_csv(filename1, index_col=0)
df2 = pd.read_csv(filename2, index_col=0)
df3 = pd.read_csv(filename3, index_col=0)
print(df1)
print(df2)
labels = range(len(list(df1.columns.values)))
rmse_per_feature_lst= []
rmse_per_feature_lst.append(df1.iloc[-1].tolist())
rmse_per_feature_lst.append(df2.iloc[-1].tolist())
rmse_per_feature_lst.append(df3.iloc[-1].tolist())
x = np.arange(len(rmse_per_feature_lst[0]))

runs = 3
fig3, ax = plt.subplots()
xmin, xmax, ymin, ymax = ax.axis()
width = ((xmax-xmin)/ len(rmse_per_feature_lst[0]))/runs * 20
for r in range(runs):
    ax.bar(x - width/2 +r*width, rmse_per_feature_lst[r], width, label=imputer_lst[r])
ax.set_xlabel("Feature")
ax.set_ylabel("RMSE")
ax.set_xticks(x)
ax.set_xticklabels(labels)
plt.title("Imputer RMSE performance comparison: "+data_name)
plt.legend()
fig3.tight_layout()
plt.show()