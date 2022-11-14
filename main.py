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

'''Main function for UCI letter and spam datasets.
'''

#python3 main.py --data_name adult  --miss_rate 0.8 --batch_size 128 --hint_rate 0.9 --alpha 100 --iterations 10000

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import mean_squared_error

from data_loader import data_loader
from gain import gain
from utils import rmse_loss

def main (args):
  '''Main function for UCI letter and spam datasets.
  
  Args:
    - data_name: letter, spam or adult
    - miss_rate: probability of missing components
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations

  Returns:
    - imputed_data_x: imputed data
    - rmse: Root Mean Squared Error
  '''
  
  data_name = args.data_name
  miss_rate = args.miss_rate
  
  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}
  
  # Load data and introduce missingness
  ori_data_x, miss_data_x, data_m = data_loader(data_name, miss_rate)
  
  loss_list = []

  """
  Different imputer selection: Simple Imputer, KNN, GAIN
  """
  # Impute missing data

  # imputed_data_x= KNNImputer()
  # imputer = SimpleImputer(strategy='constant')
  # imputed_data_x = imputer.fit_transform(miss_data_x)

  imputed_data_x, loss_list = gain(miss_data_x, gain_parameters)
  print("Loss list", loss_list)
  x = np.arange(len(loss_list))
  plt.plot(x, loss_list)
  plt.show()

  # rmse = mean_squared_error(imputed_data_x[data_m==0], ori_data_x[data_m==0], squared = False)
  # print("Rmse :", rmse)
  # Finally Report the RMSE performance
  rmse = rmse_loss(ori_data_x, imputed_data_x, data_m)
  
  print('imputed_data_x: ', imputed_data_x)
  print('RMSE Performace: ' + str(np.round(rmse, 4)))

  return imputed_data_x, rmse

if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      choices=['letter','spam', 'adult', 'Compas'],
      default='spam',
      type=str)
  parser.add_argument(
      '--miss_rate',
      help='missing data probability',
      default = 0.2,
      type=float)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch',
      default=128,
      type=int)
  parser.add_argument(
      '--hint_rate',
      help='hint probability',
      default = 0.9,
      type=float)
  parser.add_argument(
      '--alpha',
      help='hyperparameter',
      default = 100,
      type=float)
  parser.add_argument(
      '--iterations',
      help='number of training interations',
      default = 10000,
      type=int)
  
  args = parser.parse_args() 
  

  # Calls main function  
  imputed_data, rmse = main(args)
