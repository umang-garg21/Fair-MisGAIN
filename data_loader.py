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

'''
Data loader
'''

# Necessary packages
import numpy as np
import pandas as pd
import pickle as pkl
from utils import binary_sampler
from keras.datasets import mnist
from load_adult import load_adult
from load_compas import load_compas_data
from autoencoder_module import autoencoder

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

def unstack(a, axis = 0):
  return [np.squeeze(e, axis) for e in np.split(a, a.shape[axis], axis = axis)]

def data_loader (data_name, miss_rate, drop_f_lst, no_impute_f= [], sensitive_f_lst= []):
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
  binary_features = []
  sensitive_features = [] 
  continuous_col = []
  continuous_features = []

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

    with open(train_file,'w+') as file:
      file.truncate(0)
    # print("train_data.values :", train_data.values)
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)
    
    df_obj = pd.read_csv(train_file)

    # Removing columns for imputation
    if drop_f_lst:
      df_obj = df_obj.drop(df_obj.columns[drop_f_lst], axis=1)

    # Constructing categorical columns for the remaining data #
    # categorical fields
    category_col = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                    'relationship', 'native-country', 'income']
    binary_col = ['sex']
 
    for f_num, col in enumerate(list(df_obj.columns.values)):
      print("f_num, col", f_num, col)
      if col in category_col:
          categorical_features.append(f_num)
      elif col in binary_col:
          binary_features.append(f_num)
      else:
          continuous_col.append(col)
          continuous_features.append(f_num)

    with open(file_name,'r+') as file:
      file.truncate(0)
    df_obj.to_csv(file_name, index=False)
    labels = list(df_obj.columns.values)
    print("Features present in the dataset used by GAIN:", labels)

    # Find label index of the feature to not introduce missingess to.
    no_impute_f_indices = []
    for f in no_impute_f:
      no_impute_f_indices.append(labels.index(f))
    print("no_impute_f_indices", no_impute_f_indices)
    
    # Ref labels: ['age', 'workclass', 'education', 'education-num', 
    # 'marital-status', 'occupation', 'relationship', 'race', 'gender', 
    # 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

    data_x = np.loadtxt(file_name, delimiter=",", skiprows = 1)
  
  elif data_name == 'Compas':
    data_x, y, z, labels = load_compas_data()
    train_file = 'data/'+ data_name +'.csv'
    print("output vector in COMPAS", y)
    print("labels in COMPAS", labels)

    with open(train_file,'r+') as file:
      file.truncate(0)
    # print("train_data.values :", train_data.values)
    df = pd.DataFrame(data_x, columns=labels)
    df.to_csv(train_file, index=False, header=True, sep=',')
    df_obj = pd.read_csv(train_file)

    # Find label index of the feature to not introduce missingess to.
    no_impute_f_indices = []
    for f in no_impute_f:
      no_impute_f_indices.append(labels.index(f))
    print("no_impute_f_indices", no_impute_f_indices)

    ## Age category is characterized and one-hot encoded into 3 categories.
    binary_col = ['age_cat_25 - 45', 'age_cat_Greater than 45', 'age_cat_Less than 25', 'race', 'sex', 'c_charge_degree']
    category_col = []

    for f_num, col in enumerate(list(df_obj.columns.values)):
      print("f_num, col", f_num, col)
      if col in category_col:
        categorical_features.append(f_num)
      elif col in binary_col:
        binary_features.append(f_num)
      else:
        continuous_col.append(col)
        continuous_features.append(f_num)

  elif data_name == 'HSLS':
    # df = pd.read_pickle('hsls_orig.pkl')
    # df = df.replace(r'^\s*$', np.nan, regex=True)

    student_pets = pd.read_csv("data/hsls_17_student_pets_sr_v1_0.csv")
    school = pd.read_csv("data/hsls_09_school_v1_0.csv")
    
    student_vars = ['X1SEX', 'X1RACE', 'X1MTHID', 'X1MTHUTI', 'X1MTHEFF', 'X1MTHINT','X1PAR1EDU', 'X1PAR2EDU', 
                'X1PAR1OCC2', 'X1PAR2OCC2', 'X1FAMINCOME',
                'S1ENGCOMP', 'S1MTHCOMP', 'S1SCICOMP', 'S1APCALC', 'S1IBCALC']

    parent_vars = ['P1JOBNOW1', 'P1JOBONET1_STEM1', 'P1JOBONET2_STEM1','P1MTHHWEFF', 'P1SCIHWEFF', 'P1ENGHWEFF', 
               'P1MTHCOMP', 'P1SCICOMP', 'P1ENGCOMP', 'P1CAMPMS', 'P1CAMPOTH', 'P1NOOUTSCH', 'P1MUSEUM', 
               'P1COMPUTER', 'P1FIXED', 'P1SCIFAIR', 'P1SCIPROJ', 'P1STEMDISC', 'P1EDUASPIRE', 'P1EDUEXPECT']

    grade_vars = ['X1TXMSCR', 'X2TXMSCR']
    
    all_vars = grade_vars + student_vars + parent_vars

    df = student_pets[all_vars]
    df[df <= -7] = np.nan

    """ 
    Preprocessing steps for Race 
    """
    df = df.dropna()

    ## Dropping rows that are missing race or sex ##
    # df = df[df['X1RACE'].notna() & df['X1SEX'].notna() & df['X1TXMSCR'].notna()]

    ## Creating sexbin & gradebin variable (9thgrade) ## 
    df['sexbin'] = (df['X1SEX']==1).astype(int)
    df['gradebin'] = (df['X1TXMSCR'] > df['X1TXMSCR'].median()).astype(int)

    ## Dropping sex and 12th grade data just to focus on the 9th grade prediction ##
    df = df.drop(columns=['X1SEX', 'X1TXMSCR', 'X2TXMSCR'])

    ## Scaling ##
    scaler = MinMaxScaler() 
    df = pd.DataFrame(scaler.fit_transform(df) , columns=df.columns,index=df.index)

    ## Balancing data to have roughly equal race=0 and race =1 ##
    #df = balance_data(df, 'sexbin')
    df.describe()

    category_col = ['X1RACE', 'X1MTH1D', 'X1PAR1EDU', 'X1PAR2EDU', 'X1FAMINCOME', 'S1ENGCOM', 
                    'S1MTHCOMP', 'S1SCICOMP', 'S1APCALC', 'S1IBCALC', 'P1JOBONET1_STEM1',
                     'P1JOBONET2_STEM1', 'P1MTHHWEFF', 'P1SCIHWEFF', 'P1ENGHWEFF', 'P1MTHCOMP', 
                     'X1MTHID', 'X1MTHUTI', 'P1SCICOMP', 'P1ENGCOMP', 'P1EDUASPIRE',
                      'P1EDUEXPECT', 'X1PAR1OCC2', 'X1PAR2OCC2']

    binary_col = ['P1JOBNOW1', 'P1CAMPMS', 'P1CAMPOTH', 'P1NOOUTSCH', 'P1MUSEUM', 
                  'P1COMPUTER', 'P1FIXED', 'P1SCIFAIR', 'P1SCIPROJ', 'P1STEMDISC',
                  'sexbin', 'gradebin']

    ############################################################
    ##### Preprocessing steps for Sex: Uncomment to use. #######
    ############################################################

    """
    ## Dropping rows that are missing race or sex ##
    df = df[df['X1RACE'].notna() & df['X1SEX'].notna() & df['X1TXMSCR'].notna()]

    ## Creating sexbin & gradebin variable (9thgrade) ## 
    df['sexbin'] = (df['X1SEX']==1).astype(int)
    df['gradebin'] = (df['X1TXMSCR'] > df['X1TXMSCR'].median()).astype(int)

    ## Dropping sex and 12th grade data just to focus on the 9th grade prediction ##
    df = df.drop(columns=['X1SEX', 'X1TXMSCR', 'X2TXMSCR'])

    ## Scaling ##
    scaler = MinMaxScaler() 
    df = pd.DataFrame(scaler.fit_transform(df) , columns=df.columns,index=df.index)

    ## Balancing data to have roughly equal race=0 and race =1 ##
    #df = balance_data(df, 'sexbin')

    df.describe()

    category_col = []
    binary_col = [P1JOBNOW1, P1CAMPMS, P1CAMPOTH, P1NOOUTSCH, P1MUSEUM, 
                  P1COMPUTER, P1FIXED, P1SCIFAIR, P1SCIPROJ, P1STEMDISC
                  , sexbin, gradebin]
    """

    file_name = 'data/' + data_name +'_mod' +'.csv'
    with open(file_name, 'r+') as file:
      file.truncate(0)
    df.to_csv(file_name, index=False, header=True, sep=',')
    df_obj = pd.read_csv(file_name)
    print("df_obj", df_obj)
    labels = list(df_obj.columns.values)

    no_impute_f_indices = []
    for f in no_impute_f:
      no_impute_f_indices.append(labels.index(f))
    print("no_impute_f_indices", no_impute_f_indices)

    for f_num, col in enumerate(list(df_obj.columns.values)):
      print("f_num, col", f_num, col)
      if col in category_col:
        categorical_features.append(f_num)
      elif col in binary_col:
        binary_features.append(f_num)
      else:
        continuous_col.append(col)

    data_x = np.genfromtxt(file_name, delimiter=",", skip_header = 1)

  file_name = 'data/'+data_name+'.csv'
  df = pd.read_csv(file_name)

  for f_num, col in enumerate(list(df.columns.values)):
        print("f_num, col", f_num, col)
        if col in category_col:
            categorical_features.append(f_num)
        elif col in binary_col:
            binary_features.append(f_num)
        else:
            continuous_features.append(f_num)

  # Binning values for categorical features in the dataset 
  bins =[]
  if categorical_features:
    for f in categorical_features:
      temp = np.sort(np.unique(data_x[~np.isnan(data_x[:, f]), f]))
      # Bin can be equally distributed as it is supposed to be OR exactly non nan values in data_x for now.
      # a =  len(np.unique(temp))
      # bins_f = np.arange(a)
      # vector_norm = np.linalg.norm(bins_f)
      # bins_f = bins_f/ vector_norm
      # bins.append(bins_f)
      bins.append(temp)
      # print("f", f, "temp", temp)

  if categorical_features:
    for f in categorical_features:
      one_hot = pd.get_dummies(df[f])
      df = df.drop(f, axis = 1)
      df = df.join(one_hot)
    df.to_csv(file_name, index=False, header=True, sep=',')
    data_x = np.genfromtxt(file_name, delimiter=",", skip_header = 1)


  ## BEFORE INTRODUCING MISSINGNESS, we will pass it to autoencoder module
  print(data_x)
  data_x = autoencoder(data_x)
  print(data_x)

  # Parameters
  no, dim = data_x.shape
  # Introduce missing data MCAR to relevant features
  if no_impute_f_indices is None:
    data_m = binary_sampler(1-miss_rate, no, dim)
  else:
    data_m = np.ones(data_x.shape)
    for f in range(dim):
      if f not in no_impute_f_indices:
        data_m[:, f] = np.squeeze(binary_sampler(1-miss_rate, no, 1))

    for f in no_impute_f_indices:
      print("Unique mask values for no imputed feature f", f, ":", np.unique(data_m[:, f]))
  
  miss_data_x = data_x.copy()
  miss_data_x[data_m == 0] = np.nan

  # for f in range(dim):
  #   print("unique miss_Data_X values", np.unique(miss_data_x[:, f]))

  # print("Original data is :", data_x)
  # print("Data with missing entries:", miss_data_x)
  # print("Mask matrix:", data_m)
  print("Categorical features for selected data:", categorical_features)
  print("labels", labels)

 
  if sensitive_f_lst:
    for f in sensitive_f_lst:
      sensitive_features.append(labels.index(f))
      print("sensitive_features", sensitive_features)

  # Original, MCAR X, mask, label, catergorical_features, binary_features, categorical bins 
  return data_x, miss_data_x, data_m, labels, categorical_features, binary_features, sensitive_features, bins


