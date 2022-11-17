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
    - deep_analysis: True or False for Model validity
  Returns:
    - imputed_data_x: imputed data
    - rmse: Root Mean Squared Error
  '''
  
  data_name = args.data_name
  miss_rate = args.miss_rate
  deep_analysis = args.deep_analysis
  imputer_type = args.imputer_type
  drop_f_lst = args.drop_f
  runs = args.runs
  schedule = []

  print("drop_f_lst", drop_f_lst)

  print(" ####################### In-depth Analysis status: ", deep_analysis, "###########################")

  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}
  
  # Load data and introduce missingness
  ori_data_x, miss_data_x, data_m, labels, categorical_features = data_loader(data_name, miss_rate, drop_f_lst)
  no, dim = ori_data_x.shape
  
  imputed_data_x_lst =[]
  rmse_lst =[]
  rmse_it_lst = []
  rmse_per_feature_it_lst = []

  for r in range(runs):
    loss_list = []
    """
    Different imputer selection: Simple Imputer, KNN, GAIN
    """
    # Impute missing data

    if imputer_type == 'Simple':
      imputer = SimpleImputer(strategy='constant')
      imputed_data_x = imputer.fit_transform(miss_data_x)
      rmse = mean_squared_error(imputed_data_x[data_m==0], ori_data_x[data_m==0], squared = False)
      print("Rmse :", rmse)

    elif imputer_type == 'KNN':
      imputer= KNNImputer()
      imputed_data_x = imputer.fit_transform(miss_data_x)
      rmse = mean_squared_error(imputed_data_x[data_m==0], ori_data_x[data_m==0], squared = False)
      print("Rmse :", rmse)

    elif imputer_type =='Gain':
      if deep_analysis:
        print(" ----------------- In-depth Analysis mode -------------------")
        imputed_data_x, loss_list, rmse_it, rmse_per_feature_it = gain(ori_data_x, miss_data_x, gain_parameters, schedule, categorical_features, deep_analysis)
      else:
        print(" ----------------- In-depth ANALYSIS SKIPPED ---------------------")
        imputed_data_x, loss_list = gain(ori_data_x, miss_data_x, gain_parameters, schedule, categorical_features, deep_analysis)

      # print("Loss list", loss_list)s
      y = loss_list
      y1, y2, y3, y4, y5 = [], [], [], [], []
    
      for it in range(len(loss_list)):
        y1.append(y[it][0])  # D_loss
        y2.append(y[it][1])  # MSE_loss_cont
        y3.append(y[it][2])  # MSE_loss_Cat 
        y4.append(y[it][3])  # MSE_loss_total  
        y5.append(y[it][4])  # G_loss

      x = np.arange(len(loss_list))
      p1 = plt.plot(x, y1, label = 'D_loss'+'_'+str(r))
      p2 = plt.plot(x, y2, label = 'MSE_loss_cont'+'_'+str(r))
      p3 = plt.plot(x, y3, label = 'MSE_loss_cat'+'_'+str(r))
      p4 = plt.plot(x, np.sqrt(y4), label =  'Root MSE loss of generator for existing data'+'_'+str(r))
      p5 = plt.plot(x, y5, label = 'G_loss'+'_'+str(r))
      if deep_analysis:
        p6 = plt.plot(x, rmse_it, label = 'RMSE Evolution'+'_'+str(r))
      plt.legend()
      plt.show()
      
      if deep_analysis:
        if not labels:
          for i in range(dim):
            labels.append(i)
        fig2 = plt.figure()
        x = np.arange(len(rmse_it))
        plt.plot(x, rmse_per_feature_it, label = labels)
        plt.legend()
        plt.show()
    
      # Finally Report the RMSE performance
      rmse, rmse_per_feature = rmse_loss(ori_data_x, imputed_data_x, data_m)
      
      print('imputed_data_x: ', imputed_data_x)
      print('Original data:', ori_data_x)
      print('RMSE Performace: ' + str(np.round(rmse, 4)))
      print('RMSE per feature:' + str(np.round(rmse_per_feature, 4)))
      
      fig = plt.figure()
      # creating the bar plot
      if labels:
        x = labels 
      else:
        x = np.arange(len(rmse_per_feature)) 
      plt.bar(x, rmse_per_feature, label='run'+str(r))
      plt.xlabel("Feature")
      plt.ylabel("RMSE")
      plt.title("RMSE per feature")
      plt.legend()
      plt.show()

    imputed_data_x_lst.append(imputed_data_x)
    rmse_lst.append(rmse)

  return imputed_data_x_lst, rmse_lst

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
      '--deep_analysis',
      help='Deeper Analysis for model validity',
      choices= [True, False],
      default = False,
      type=bool)
  parser.add_argument(
      '--imputer_type',
      help='Select imputer',
      choices = ['Simple', 'KNN', 'Gain'],
      default = 'Gain',
      type=str)
  parser.add_argument(
      '--iterations',
      help='number of training interations',
      default = 10000,
      type=int)
  parser.add_argument(
      '-drop_f','--drop_f',
      help= 'features to be dropped',
      nargs='*',
      type = int)
  parser.add_argument(
      '-runs', '--runs',
      help= 'Give number of runs for performance analysis',
      default = 1,
      type = int)

  args = parser.parse_args() 
  
  print(args)
  # Calls main function  
  imputed_data_lst, rmse_lst = main(args)
