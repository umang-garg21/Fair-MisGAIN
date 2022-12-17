import matplotlib.pyplot as plt
import numpy as np

rmse_per_feature_lst= []
# KNN
rmse_per_feature_lst.append([0.2574, 0.1895, 0.2581, 0.2144, 0.2566, 0.3148, 0.3415, 0.2703, 0.5659, 0.0772, 0.112,  0.1415, 0.1547])
# GAIN
rmse_per_feature_lst.append([0.1953, 0.175,  0.2662, 0.1766, 0.2699, 0.3196, 0.3287, 0.2714, 0.5717, 0.139, 0.1024, 0.2557, 0.1617])
x = np.arange(len(rmse_per_feature_lst[0]))
labels = ["age", "workclass", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
            "hours-per-week", "native-country"]
runs = 2
fig3, ax = plt.subplots()
xmin, xmax, ymin, ymax = ax.axis()
width = ((xmax-xmin)/ len(rmse_per_feature_lst[0]))/runs *10
for r in range(runs):
    ax.bar(x - width/2 +r*width, rmse_per_feature_lst[r], width, label='run'+str(r))
ax.set_xlabel("Feature")
ax.set_ylabel("RMSE")
ax.set_xticks(x)
ax.set_xticklabels(labels) 
plt.title("Imputer RMSE performance comparison")
#plt.legend('GAIN', 'KNN')
fig3.tight_layout()
plt.show()