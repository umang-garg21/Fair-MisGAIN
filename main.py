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
##### CMD line input example ######################
# python main.py --data_name adult  --miss_rate 0.1 --batch_size 128 --hint_rate 0.9  --alpha 100 --iterations 1000  --runs 5 --drop_f 8  --imputer Gain --deep_analysis False -bin_category_f True -use_cont_f True -use_cat_f True

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import matplotlib.pyplot as plt
from labellines import labelLine, labelLines
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import mean_squared_error

from data_loader import data_loader
from gain import gain
from utils import rmse_loss, normalization, renormalization

def parse_boolean(value):
    value = value.lower()

    if value in ["true", "yes", "y", "1", "t"]:
        return True
    elif value in ["false", "no", "n", "0", "f"]:
        return False

    return False

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
    - imputer type: select imputer 
    - drop_f: option to drop features
    - runs: number of runs

  Returns:
    - imputed_data_x_lst: imputed data list for all runs
    - rmse_lst: Root Mean Squared Error list for all runs
  '''
  
  data_name = args.data_name
  miss_rate = args.miss_rate
  deep_analysis = args.deep_analysis
  imputer_type = args.imputer_type
  drop_f_lst = args.drop_f
  runs = args.runs
  bin_category_f = args.bin_category_f
  use_cont_f = args.use_cont_f
  use_cat_f = args.use_cat_f
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
  rmse_per_feature_lst =[]
  rmse_it_lst = []
  rmse_per_feature_it_lst = []
  
  ylst = [[] for i in range(runs)]

  for r in range(runs):
    loss_list = []
    """
    Different imputer selection: Simple Imputer, KNN, GAIN
    """
    # Impute missing data 
    if imputer_type == 'Simple':
      imputer = SimpleImputer(strategy='constant')
      imputed_data_x = imputer.fit_transform(miss_data_x)
      deep_analysis = False
      
      """
      ori_data_x, norm_parameters = normalization(ori_data_x)
      imputed_data_x, _ = normalization(imputed_data_x, norm_parameters)
      rmse_org = mean_squared_error(ori_data_x[data_m==0], imputed_data_x[data_m ==0], squared =False) 
      # rmse_1 = np.sqrt(np.square(np.subtract(np.array([1, 0, 1]),np.array([1, 0, 0]))).mean())
      # print("RMSE_1", rmse_1)
      print("RMSE original: ", rmse_org)
      """
      
    elif imputer_type == 'KNN':
      imputer= KNNImputer()
      imputed_data_x = imputer.fit_transform(miss_data_x)
      deep_analysis = False
      """
      ori_data_x, norm_parameters = normalization(ori_data_x)
      imputed_data_x, _ = normalization(imputed_data_x, norm_parameters)
      rmse_org = mean_squared_error(ori_data_x[data_m==0], imputed_data_x[data_m ==0], squared =False) 
      # rmse_1 = np.sqrt(np.square(np.subtract(np.array([1, 0, 1]),np.array([1, 0, 0]))).mean())
      # print("RMSE_1", rmse_1)
      print("RMSE original: ", rmse_org)
      """

    elif imputer_type =='Gain':
      if deep_analysis:
        print(" ----------------- In-depth Analysis mode -------------------")
        imputed_data_x, loss_list, rmse_it, rmse_per_feature_it = gain(ori_data_x, miss_data_x, gain_parameters, schedule, categorical_features, deep_analysis, bin_category_f, use_cont_f, use_cat_f)
      else:
        print(" ----------------- In-depth analysis skipped ---------------------")
        imputed_data_x, loss_list = gain(ori_data_x, miss_data_x, gain_parameters, schedule, categorical_features, deep_analysis, bin_category_f, use_cont_f, use_cat_f)
    
      # print("Loss list", loss_list)s
      y = loss_list
      ylst[r] = loss_list
      y1, y2, y3, y4, y5 = [], [], [], [], []
          
      if runs == 1:
        for it in range(len(loss_list)):
          y1.append(y[it][0])  # D_loss
          y2.append(y[it][1])  # MSE_loss_cont
          y3.append(y[it][2])  # MSE_loss_Cat 
          y4.append(y[it][3])  # MSE_loss_total  
          y5.append(y[it][4])  # G_loss

        x = np.arange(len(loss_list))
        fig1 = plt.figure()
        p1 = plt.plot(x, y1, label = 'D_loss'+'_'+str(r))
        p2 = plt.plot(x, y2, label = 'MSE_loss_cont'+'_'+str(r))
        p3 = plt.plot(x, y3, label = 'MSE_loss_cat'+'_'+str(r))
        p4 = plt.plot(x, np.sqrt(y4), label =  'Root MSE loss of generator for existing data'+'_'+str(r))
        p5 = plt.plot(x, y5, label = 'G_loss'+'_'+str(r))
        if deep_analysis:
          p6 = plt.plot(x, rmse_it, label = 'RMSE Evolution'+'_'+str(r))
        plt.legend()
        labelLines(plt.gca().get_lines(), align=False)
        plt.show()

      
      if deep_analysis:
        if not labels:
          for i in range(dim):
            labels.append(i)    
        if runs == 1:
          fig2 = plt.figure()
          x = np.arange(len(rmse_it))
          plt.plot(x, rmse_per_feature_it, label = labels)
          # plt.legend()
          labelLines(plt.gca().get_lines(), align=False)
          plt.show()
      
    # Finally Report the RMSE performance

    rmse, rmse_per_feature = rmse_loss(ori_data_x, imputed_data_x, data_m)
    rmse_lst.append(rmse)
    rmse_per_feature_lst.append(rmse_per_feature)
    # print('imputed_data_x: ', imputed_data_x)
    # print('Original data:', ori_data_x)
    print('RMSE Performace_' +'run_'+str(r)+ ": " +str(np.round(rmse, 4)))
    print('RMSE per feature_' +'run_'+str(r)+ str(np.round(rmse_per_feature, 4)))

    imputed_data_x_lst.append(imputed_data_x)
    rmse_lst.append(rmse)
    if deep_analysis:
      rmse_it_lst.append(rmse_it)
      rmse_per_feature_it_lst.append(rmse_per_feature_it)

  # creating the bar plot for RMSE per feature plot for all runs
  x = np.arange(len(rmse_per_feature))
  if runs == 1:
    fig3= plt.figure()
    ax = plt.gca()
    plt.bar(x, rmse_per_feature, label='run'+str(r))
  else:
    fig3, ax = plt.subplots()
    xmin, xmax, ymin, ymax = ax.axis()
    width = ((xmax-xmin)/ len(rmse_per_feature))/runs *10
    for r in range(runs):
      ax.bar(x - width/2 +r*width, rmse_per_feature_lst[r], width, label='run'+str(r))
  ax.set_xlabel("Feature")
  ax.set_ylabel("RMSE")
  ax.set_xticks(x)
  ax.set_xticklabels(labels) 
  plt.title(imputer_type+" imputer: RMSE per feature ")
  fig3.tight_layout()
  plt.show()

  # Plot RMSE for all runs
  if runs==1:
    pass
  else:
    fig4 = plt.figure()
    plt.plot(np.arange(len(rmse_lst)), rmse_lst)
    plt.title(imputer_type+" imputer: RMSE")
    plt.legend()
    plt.show()
    
  return imputed_data_x_lst, rmse_lst, rmse_it_lst, rmse_per_feature_it_lst, ylst

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
      help= 'features to be dropped..Categorical/ Continuous data can be toggled for gain in gain.py',
      nargs='*',
      default = None,
      type = int)
  parser.add_argument(
      '-runs', '--runs',
      help= 'Give number of runs for performance analysis',
      default = 1,
      type = int)
  parser.add_argument(
      '--deep_analysis',
      help='Deeper Analysis for model validity',
      choices= [True, False],
      default = False,
      type=parse_boolean)
  parser.add_argument(
      '-bin_category_f', '--bin_category_f',
      help= 'toggle binning category features',
      default = False,
      type = parse_boolean)
  parser.add_argument(
      '-use_cat_f', '--use_cat_f',
      help= 'use categorical features for training?',
      default = True,
      type = parse_boolean)
  parser.add_argument(
      '-use_cont_f', '--use_cont_f',
      help= 'use continuous features for training?',
      default = True,
      type = parse_boolean)

  args = parser.parse_args() 
  
  print(args)
  # Calls main function  
  imputed_data_lst, rmse_lst, rmse_it_lst, rmse_per_feature_it_lst, ylst = main(args)
