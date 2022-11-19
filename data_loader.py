# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Data loader for UCI letter, spam and MNIST datasets.
'''

# Necessary packages
import numpy as np
import pandas as pd
from utils import binary_sampler
from keras.datasets import mnist
from load_adult import load_adult

def data_loader (data_name, miss_rate, drop_f_lst):
  '''Loads datasets and introduce missingness.
  
  Args:
    - data_name: letter, spam, or mnist
    - miss_rate: the probability of missing components
    
  Returns:
    data_x: original data
    miss_data_x: data with missing values
    data_m: indicator matrix for missing components
  '''

  # Load data
  labels = []  
  categorical_features = []
  if data_name in ['letter', 'spam']:
    file_name = 'data/'+data_name+'.csv'
    data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)   

  elif data_name == 'mnist': 
    (data_x, _), _ = mnist.load_data()
    data_x = np.reshape(np.asarray(data_x), [60000, 28*28]).astype(float)

  elif data_name == 'adult': 
    smaller = False
    scalar = True
    # a is train, b is test
    train_data, test_data = load_adult(smaller, scalar, miss_rate)  # currently drop_p is not used, used below instead
    file_name = 'data/'+data_name+'.csv'
    train_file = 'data/'+'train_'+data_name+'.csv'
    test_file = 'data/'+'test_'+data_name+'.csv'

    with open(train_file,'r+') as file:
      file.truncate(0)
    with open(test_file,'r+') as file:
      file.truncate(0)
    # print("train_data.values :", train_data.values)
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)
    
    df_obj = pd.read_csv(train_file)

    #Removing columns for imputation: check with and without #
    if drop_f_lst:
      df_obj = df_obj.drop(df_obj.columns[drop_f_lst], axis=1)

    # Constructing categorical columns for the remaining data #
    # categorical fields
    category_col = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                    'relationship', 'sex', 'native-country', 'income']

    categorical_features = []
    for f_num, col in enumerate(list(df_obj.columns.values)):
        if col in category_col:
            categorical_features.append(f_num)

    with open(file_name,'r+') as file:
      file.truncate(0)
    df_obj.to_csv(file_name, index=False)
    labels = list(df_obj.columns.values)

    print("labels for Imputation (Removing income label):", labels)
    # Ref labels: ['age', 'workclass', 'education', 'education-num', 
    # 'marital-status', 'occupation', 'relationship', 'race', 'gender', 
    # 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

    data_x = np.loadtxt(file_name, delimiter=",", skiprows = 1)
    
  elif data_name == 'Compas':
    pass

  # Parameters
  no, dim = data_x.shape
  
  # Introduce missing data MCAR
  data_m = binary_sampler(1-miss_rate, no, dim)

  miss_data_x = data_x.copy()
  miss_data_x[data_m == 0] = np.nan

  # print("Original data is :", data_x)
  # print("Data with missing entries:", miss_data_x)
  # print("Mask matrix:", data_m)
  print("Categorical features for selected data:", categorical_features)

  return data_x, miss_data_x, data_m, labels, categorical_features  # Original, MCAR X, mask, label, catergorical_feature 


