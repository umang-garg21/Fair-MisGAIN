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
python main.py --data_name adult  --miss_rate 0.1 --batch_size 128 --hint_rate 0.9 --alpha 10 --iterations 200 --runs 1 --drop_f --imputer Gain --deep_analysis True -bin_category_f True -use_cont_f True  -use_cat_f True  --no_impute_f income -sensitive_f_lst sex
python main.py --data_name Compas  --miss_rate 0.1 --batch_size 128 --hint_rate 0.9 --alpha 10 --iterations 200  --runs 1 --drop_f  --imputer Gain --deep_analysis True -bin_category_f True -use_cont_f True  -use_cat_f True  --no_impute_f -sensitive_f_lst sex
python main.py --data_name HSLS  --miss_rate 0.1 --batch_size 128 --hint_rate 0.9 --alpha 10 --iterations 200  --runs 1 --drop_f  --imputer Gain --deep_analysis True -bin_category_f True -use_cont_f True  -use_cat_f True  --no_impute_f -sensitive_f_lst sexbin
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import csv as csv
import os
import matplotlib.pyplot as plt
from labellines import labelLine, labelLines
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics

from data_loader import data_loader
from gain import gain
from utils import rmse_loss, normalization, renormalization, digitizing, ROC_Analysis

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
    - data_name: letter, spam, adult, HSLS, Compas
    - miss_rate: probability of missing components
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations
    - deep_analysis: True or False for Model validity
    - imputer type: select imputer 
    - drop_f: option to drop features
    - runs: number of runs
    - no_impute_f: feature list not to introduce missigness.
    - use_cont_f : use continuous features for training
    - use_cat_f: use catergorical features for training

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
  no_impute_f = args.no_impute_f
  sensitive_f_lst = args.sensitive_f_lst
  schedule = []

  print("drop_f_lst", drop_f_lst)

  print(" ####################### In-depth Analysis status: ", deep_analysis, "###########################")

  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}
  
# Load data and introduce missingness
  
  ori_data_x, miss_data_x, data_m, labels, categorical_features, binary_features, sensitive_features, bins = data_loader(data_name, miss_rate, drop_f_lst, no_impute_f, sensitive_f_lst)
  no, dim = ori_data_x.shape 


# Prepare Train and test data
  train_sr = 0.7
  val_sr = 0.1
  test_sr = 0.2
  tmp = [i for i in range(no)]
  np.random.shuffle(tmp)
  train_indices, val_indices, test_indices = tmp[:int(train_sr*no)], tmp[int(train_sr*no):int((train_sr+val_sr)*no)], tmp[int(train_sr*no):int((train_sr+val_sr)*no):]
  train_ori_data_x = np.zeros((len(train_indices), dim))
  train_miss_data_x = np.zeros((len(train_indices), dim))
  val_ori_data_x = np.zeros((len(val_indices), dim))
  val_miss_data_x = np.zeros((len(val_indices), dim))
  test_ori_data_x = np.zeros((len(test_indices), dim))
  test_miss_data_x = np.zeros((len(test_indices), dim))
  
  for i, ele in enumerate(train_indices):
    train_ori_data_x[i, :] = ori_data_x[ele,:]
    train_miss_data_x[i, :]= miss_data_x[ele, :]
  for i, ele in enumerate(val_indices):
    val_ori_data_x[i, :]= ori_data_x[ele,:]
    val_miss_data_x[i, :] = miss_data_x[ele, :]
  for i, ele in enumerate(test_indices):
    test_ori_data_x[i, :] = ori_data_x[ele,:]
    test_miss_data_x[i, :] = miss_data_x[ele, :]

  imputed_data_x_lst =[]
  rmse_lst =[]
  rmse_per_feature_lst =[]
  rmse_it_lst = []
  rmse_per_feature_it_lst = []

  ylst = [[] for i in range(runs)]

  for r in range(runs):
    loss_list = []
    """
    Different imputer selection: Simple Imputer, KNN, GAIN, MICE
    """
    # Impute missing data 
    if imputer_type == 'Simple':
      imputer = SimpleImputer(strategy='constant')
      imputed_data_x = imputer.fit_transform(miss_data_x)
      imputed_data_x_out = digitizing(imputed_data_x, ori_data_x, bins, categorical_features, binary_features, 0.5)
      deep_analysis = False

    elif imputer_type == 'KNN':
      imputer= KNNImputer()
      imputed_data_x = imputer.fit_transform(miss_data_x)
      imputed_data_x_out = digitizing(imputed_data_x, ori_data_x, bins, categorical_features, binary_features, 0.5)
      deep_analysis = False
        
    elif imputer_type == 'MICE':
      lr = LinearRegression()
      imputer = IterativeImputer(estimator=lr, verbose=2, max_iter=10, tol=1e-10, imputation_order='roman', sample_posterior=False)
      imputed_data_x = imputer.fit_transform(miss_data_x) 
      imputed_data_x_out = digitizing(imputed_data_x, ori_data_x, bins, categorical_features, binary_features, 0.5)
      deep_analysis = False

    elif imputer_type =='Gain':
      # Train the network
      if deep_analysis:
        print(" ----------------- In-depth Analysis mode -------------------")
        imputed_data_x_out, loss_list, rmse_it, rmse_per_feature_it, fpr, tpr = gain(train_ori_data_x, train_miss_data_x, val_ori_data_x, val_miss_data_x, test_ori_data_x, test_miss_data_x, gain_parameters, schedule, categorical_features, binary_features, sensitive_features, bins, deep_analysis, bin_category_f, use_cont_f, use_cat_f)
      else:
        print(" ----------------- In-depth analysis skipped ---------------------")
        imputed_data_x_out, loss_list, fpr, tpr = gain(train_ori_data_x, train_miss_data_x, val_ori_data_x, val_miss_data_x, test_ori_data_x, test_miss_data_x, gain_parameters, schedule, categorical_features, binary_features, sensitive_features, bins, deep_analysis, bin_category_f, use_cont_f, use_cat_f)

      for f in range(0, len(labels)):
        print("Unique feature values for feature ",f, "in initial data:", np.unique(miss_data_x[:, f]))
        print(labels[f])
        print("Unique feature values for feature",f, ":", "in imputed data", np.unique(imputed_data_x_out[:, f]))
        # print("Loss list", loss_list)
      y = loss_list
      ylst[r] = loss_list
      y1, y2, y3, y4, y5 = [], [], [], [], []
      
      for it in range(len(loss_list)):
        y1.append(y[it][0])  # D_loss
        y2.append(y[it][1])  # MSE_loss_not_binary
        y3.append(y[it][2])  # MSE_loss_binary
        y4.append(y[it][3])  # MSE_loss_total  
        y5.append(y[it][4])  # G_loss

      x = np.arange(len(loss_list))
      fig1 = plt.figure()
      plt.plot(x, y1, label = 'D_loss'+'_'+str(r))
      plt.plot(x, y2, label = 'MSE_loss_non_binary'+'_'+str(r))
      plt.plot(x, y3, label = 'MSE_loss_binary'+'_'+str(r))
      plt.plot(x, y4, label =  'MSE loss of generator for existing data'+'_'+str(r))
      plt.plot(x, y5, label = 'G_loss'+'_'+str(r))
      if deep_analysis:
        plt.plot(x, rmse_it, label = 'RMSE Evolution'+'_'+str(r))
      plt.legend()
      # labelLines(plt.gca().get_lines(), align=False)
    
      if deep_analysis:
        if not labels:
          for i in range(dim):
            labels.append(i)    
        if runs == 1:
          fig2 = plt.figure()
          x = np.arange(len(rmse_it))
          plt.plot(x, rmse_per_feature_it, label = labels)
          plt.legend()
          #labelLines(plt.gca().get_lines(), align=False)
          plt.show()
    
    # Finally Report the RMSE performance
    if imputer_type == 'Gain':
      rmse, rmse_per_feature = rmse_loss(test_ori_data_x, imputed_data_x_out, 1-np.isnan(test_miss_data_x), categorical_features, binary_features)
    else:
      rmse, rmse_per_feature = rmse_loss(ori_data_x, imputed_data_x_out, data_m, categorical_features, binary_features)

    rmse_lst.append(rmse)
    rmse_per_feature_lst.append(rmse_per_feature)
    # print('imputed_data_x_out: ', imputed_data_x_out)
    # print('Original data:', ori_data_x)
    print('RMSE Performace_' +'run_'+str(r)+ ": " +str(np.round(rmse, 4)))
    print('RMSE per feature_' +'run_'+str(r)+ str(np.round(rmse_per_feature, 4)))

    imputed_data_x_lst.append(imputed_data_x_out)
    rmse_lst.append(rmse)
    if deep_analysis:
      rmse_it_lst.append(rmse_it)
      rmse_per_feature_it_lst.append(rmse_per_feature_it)

    ##################################################################
    ###### Analysis ROC and RMSE of all features for sensitive groups ########
    ##################################################################    
    for f in sensitive_features:
      print("Sensitive features used", f)
      fpr_g1 = {}
      tpr_g1= {}
      fpr_g2 = {}
      tpr_g2 = {}
      fpr = {}
      tpr = {}
      rmse_sensitivity = {}
      test_data_m = 1 - np.isnan(test_miss_data_x)
      g1_ori_data_x= test_ori_data_x[test_ori_data_x[:, f]==0]
      g1_imputed_data_x= imputed_data_x_out[test_ori_data_x[:, f]==0]
      g1_data_m = test_data_m[test_ori_data_x[:, f]==0]
      g2_ori_data_x= test_ori_data_x[test_ori_data_x[:, f]==1]
      g2_imputed_data_x= imputed_data_x_out[test_ori_data_x[:, f]==1]
      g2_data_m = test_data_m[test_ori_data_x[:, f]==1]
      
      fpr_g1, tpr_g1 = ROC_Analysis(g1_imputed_data_x, g1_ori_data_x, bins, categorical_features, binary_features, sensitive_features)
      fpr_g2, tpr_g2 = ROC_Analysis(g2_imputed_data_x, g2_ori_data_x, bins, categorical_features, binary_features, sensitive_features)

      for i in range(dim):
        if i not in sensitive_features:
          if i in binary_features:
            fpr[i] = [abs(fpr_g1[i][j] - fpr_g2[i][j]) for j in range(len(fpr_g1[i]))]
            tpr[i] = [abs(tpr_g1[i][j] - tpr_g2[i][j]) for j in range(len(tpr_g1[i]))]
          else:
            rmse_g1, rmse_per_feature_g1 = rmse_loss(g1_ori_data_x, g1_imputed_data_x, g1_data_m, categorical_features, binary_features)
            rmse_g2, rmse_per_feature_g2 = rmse_loss(g2_ori_data_x, g2_imputed_data_x, g2_data_m, categorical_features, binary_features)
            rmse_sensitivity[i] = abs(rmse_per_feature_g1[i] - rmse_per_feature_g2[i])
      print("rmse_per_feature_g1", rmse_per_feature_g1)
      print("rmse_per_feature_g2", rmse_per_feature_g2)
      for f in binary_features:
          rmse_per_feature_g1[f] = 0
          rmse_per_feature_g2[f] = 0
          rmse_sensitivity[f] = 0

      rmse_per_feature_glist = [rmse_per_feature_g1, rmse_per_feature_g2]
      print("FPR_g1", fpr_g1)
      print("FPR_g2", fpr_g2)
      print("TPR_G1", tpr_g1)
      print("TPR_g2", tpr_g2)
      # print("RMSE sensitivity", rmse_sensitivity)
      # print("False Posiitive rate:", fpr)
      # print("True Positive Rate:", tpr)

  ####### creating the bar plot for RMSE per feature plot for all runs #######
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
  plt.title(imputer_type+" imputer: RMSE per feature")
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

  ####### creating the bar plot for RMSE per feature plot grouped by sensitive features #######
  if sensitive_features:
    for f in sensitive_features:
      x = np.arange(len(rmse_per_feature_g1))
      runs = len(np.unique(test_ori_data_x[:, f]))
      fig3, ax = plt.subplots()
      xmin, xmax, ymin, ymax = ax.axis()
      width = ((xmax-xmin)/ len(rmse_per_feature_g1))/runs *20
      for r in range(runs):
        ax.bar(x - width/2 +r*width, rmse_per_feature_glist[r], width, label='Group'+str(r))
      ax.set_xlabel("Feature")
      ax.set_ylabel("RMSE")
      ax.set_xticks(x)
      ax.set_xticklabels(labels)
      plt.title(imputer_type+" imputer: RMSE per feature ")
      fig3.tight_layout()
      plt.show()

  header = labels
  filename = 'Results//Imputer_RMSE/Imputer_RMSE'+'_'+imputer_type+'_'+data_name+'.csv'
  path = './'+filename
  if not os.path.isfile(path):
    with open(filename,'w+') as file:
      writer = csv.writer(file)
      writer.writerow(header)
  with open(filename,'a+') as file:
    r = rmse_per_feature_lst[0]
    writer = csv.writer(file)
    writer.writerow(r)
  
  return imputed_data_x_lst, rmse_lst, rmse_it_lst, rmse_per_feature_it_lst, ylst

if __name__ == '__main__':  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      choices=['letter','spam', 'adult', 'Compas', 'HSLS'],
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
      choices = ['Simple', 'KNN', 'Gain', 'MICE'],
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
      '-no_impute_f','--no_impute_f',
      help= 'features to not introduce missingness to.',
      nargs='*',
      default = None,
      type = str)
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
  parser.add_argument(
      '-sensitive_f_lst', '--sensitive_f_lst',
      help= 'list of sensitive features',
      nargs='*',
      default = None,
      type = str)

  args = parser.parse_args() 
  print(args)

  # Calls main function  
  imputed_data_lst, rmse_lst, rmse_it_lst, rmse_per_feature_it_lst, ylst = main(args)
