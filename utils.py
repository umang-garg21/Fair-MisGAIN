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

'''Utility functions for GAIN.

(1) normalization: MinMax Normalizer
(2) renormalization: Recover the data from normalzied data
(3) rounding: Handlecategorical variables after imputation
(4) rmse_loss: Evaluate imputed data in terms of RMSE
(5) xavier_init: Xavier initialization
(6) binary_sampler: sample binary random variables
(7) uniform_sampler: sample uniform random variables
(8) sample_batch_index: sample random batch index
'''
 
# Necessary packages
import numpy as np
#import tensorflow as tf
##IF USING TF 2 use following import to still use TF < 2.0 Functionalities
import tensorflow.compat.v1 as tf
from sklearn.metrics import mean_squared_error
tf.disable_v2_behavior()

def normalization (data, parameters=None):
  '''Normalize data in [0, 1] range.

  Args:
    - data: original data
  
  Returns
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  '''

  # Parameters
  # print("data.shape:", data.shape)
  _, dim = data.shape
  norm_data = data.copy()
  
  if parameters is None:
    # MinMax normalization
    min_val = np.zeros(dim)
    max_val = np.zeros(dim)
  
    # For each dimension
    for i in range(dim):
      min_val[i] = np.nanmin(norm_data[:, i])
      norm_data[:, i] = norm_data[:, i] - np.nanmin(norm_data[:, i])
      max_val[i] = np.nanmax(norm_data[:, i])
      norm_data[:, i] = norm_data[:, i] / (np.nanmax(norm_data[:, i]) + 1e-6)  #in case nanmax returns nan or 0, add 1e-6
      
    # Return norm_parameters for renormalization
    norm_parameters = {'min_val': min_val,
                       'max_val': max_val}

  else:
    min_val = parameters['min_val']
    max_val = parameters['max_val']
    
    # For each dimension
    for i in range(dim):
      norm_data[:,i] = norm_data[:,i] - min_val[i]
      norm_data[:,i] = norm_data[:,i] / (max_val[i] + 1e-6)  
      
    norm_parameters = parameters
      
  return norm_data, norm_parameters


def renormalization (norm_data, norm_parameters):
  '''Renormalize data from [0, 1] range to the original range.
  
  Args:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  
  Returns:
    - renorm_data: renormalized original data
  '''
  
  min_val = norm_parameters['min_val']
  max_val = norm_parameters['max_val']

  _, dim = norm_data.shape
  renorm_data = norm_data.copy()
    
  for i in range(dim):
    renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)   
    renorm_data[:,i] = renorm_data[:,i] + min_val[i]
    
  return renorm_data

def rounding (imputed_data, data_x, categorical_features =[]):
  '''Round imputed data for categorical variables.
  
  Args:
    - imputed_data: imputed data
    - data_x: original data with missing values
    
  Returns:
    - rounded_data: rounded imputed data
  '''
  
  _, dim = data_x.shape
  rounded_data = imputed_data.copy()

  """
  for i in range(dim):
    temp = data_x[~np.isnan(data_x[:, i]), i]
    # Only for the categorical variable: a makeshift way to identify categorical entries
    if len(np.unique(temp)) < 20:
      rounded_data[:, i] = np.round(rounded_data[:, i])
  """

  for f in categorical_features:
      rounded_data[:, f] = np.round(rounded_data[:, f])
      print("Orig data for feature",f, ":", np.unique(data_x[:,f]))
      print("Imputed data for feature",f, ":", np.unique(imputed_data[:,f]))
      print("Rounded data for feature",f, ":", np.unique(rounded_data[:,f]))
  
  return rounded_data


def digitizing (imputed_data, data_x, bins, categorical_features =[], binary_features=[]):
  '''
  Args:
    - imputed_data: imputed data
    - data_x: original data with missing values
    
  Returns:
    - rounded_data: rounded imputed data
  '''
  _, dim = data_x.shape
  dig_data = imputed_data.copy()

  for i, f in enumerate(categorical_features):
    # print("dig data", dig_data[:, f])
    if f not in binary_features:
      dig_data[:, f] = np.digitize(dig_data[:, f], bins[i])
      temp, norm_p = normalization(np.expand_dims(dig_data[:, f], axis=-1))
      dig_data[:, f] = np.squeeze(temp)
    else:
      dig_data[:, f] = np.round(dig_data[:, f])

  for i, f in enumerate(categorical_features):
    # print("Unq original data for feature", f, ":", np.unique(data_x[:,f]))
    # print("Bins for feature", f, ":", bins[i])
    # print("Unq Digitized data for feature",f, ":", np.unique(dig_data[:,f]))
    pass

  return dig_data


def rmse_loss (ori_data, imputed_data, data_m):
  
  '''Compute RMSE loss between ori_data and imputed_data
  
  Args:
    - ori_data: original data without missing values
    - imputed_data: imputed data
    - data_m: indicator matrix for missingness
    
  Returns:
    - rmse: Root Mean Squared Error
  '''
  # Normalize to same base min and max values to get correct MSE, RMSE
  ori_data, norm_parameters = normalization(ori_data)
  imputed_data, _ = normalization(imputed_data, norm_parameters)
  
  # print(ori_data, imputed_data)
  # Only for missing values

  MSE_1 = np.square(np.subtract(imputed_data[data_m==0], ori_data[data_m==0])).mean()
  rmse = np.sqrt(MSE_1)
  nominator = np.sum(((1-data_m) * ori_data - (1-data_m) * imputed_data)**2)  #1-data_m is 1 only when data_m is 0: imputed data
  denominator = np.sum(1-data_m)
  # rmse = np.sqrt(nominator/float(denominator))

  # Calculate RMSE per feature
  _, dim = ori_data.shape

  rmse_per_feature = []
  for i in range(dim):    
    nominator = np.sum(((1-data_m[:, i]) * ori_data[:, i] - (1-data_m[:, i]) * imputed_data[:, i])**2)  #1-data_m is 1 only when data_m is 0: imputed data
    denominator = np.sum(1-data_m[:, i])
    rmse_per_feature.append(np.sqrt(nominator/float(denominator)))

  return rmse, rmse_per_feature


def xavier_init(size):
  '''Xavier initialization.
  
  Args:
    - size: vector size
    
  Returns:
    - initialized random vector.
  '''
  in_dim = size[0]
  print("in_dim", in_dim)
  xavier_stddev = 1. / tf.sqrt(int(in_dim) / 2.)
  return tf.random_normal(shape = size, stddev = xavier_stddev)
      

def binary_sampler(p, rows, cols):
  '''Sample binary random variables.
  
  Args:
    - p: probability of 1
    - rows: the number of rows
    - cols: the number of columns
    
  Returns:
    - binary_random_matrix: generated binary random matrix.
  '''
  unif_random_matrix = np.random.uniform(0., 1., size = [rows, cols])
  binary_random_matrix = 1*(unif_random_matrix < p)
  return binary_random_matrix


def uniform_sampler(low, high, rows, cols):
  '''Sample uniform random variables.
  
  Args:
    - low: low limit
    - high: high limit
    - rows: the number of rows
    - cols: the number of columns
    
  Returns:
    - uniform_random_matrix: generated uniform random matrix.
  '''
  return np.random.uniform(low, high, size = [rows, cols])       


def sample_batch_index(total, batch_size):
  '''Sample index of the mini-batch.
  
  Args:
    - total: total number of samples
    - batch_size: batch size
    
  Returns:
    - batch_idx: batch index
  '''
  total_idx = np.random.permutation(total)
  batch_idx = total_idx[:batch_size]
  return batch_idx
  
