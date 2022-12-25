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

'''GAIN function.
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data 
           Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
Contact: jsyoon0823@gmail.com
'''

# Necessary packages
#import tensorflow as tf
##IF USING TF 2 use following import to still use TF < 2.0 Functionalities
import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import numpy as np
from tqdm import tqdm
import pandas as pd 
from scipy.stats import chi2_contingency

from utils import normalization, renormalization, rounding, digitizing
from utils import xavier_init
from utils import binary_sampler, uniform_sampler, sample_batch_index
from utils import rmse_loss


def gain (ori_data_x, data_x, gain_parameters, schedule, categorical_features=[], binary_features=[], deep_analysis=False, bin_category_f= False, use_cont_f=True, use_cat_f=True):
  '''Impute missing values in data_x
  
  Args:
    - data_x: original data with missing values
    - gain_parameters: GAIN network parameters:
      - batch_size: Batch size
      - hint_rate: Hint rate
      - alpha: Hyperparameter
      - iterations: Iterations
      
  Returns:
    - imputed_data: imputed data
  '''

  # Define mask matrix
  data_m = 1-np.isnan(data_x)
  
  # System parameters
  batch_size = gain_parameters['batch_size']
  hint_rate = gain_parameters['hint_rate']
  alpha = gain_parameters['alpha']
  iterations = gain_parameters['iterations']
 
  print("use_cat_f", use_cat_f)
  print("use_cont_f", use_cont_f)

 # Other parameters
  no, dim = data_x.shape
  # Hidden state dimensions
  h_dim = int(dim)
  G_sample_bin_correlation = []

  # Normalization
  norm_data, norm_parameters = normalization(data_x)
  norm_data_x = np.nan_to_num(norm_data, 0)
  
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
      
  ## GAIN architecture
  # Input placeholders
  # Data vector
  X = tf.placeholder(tf.float32, shape = [None, dim])
  # Mask vector
  M = tf.placeholder(tf.float32, shape = [None, dim])
  # Hint vector
  H = tf.placeholder(tf.float32, shape = [None, dim])
  
  # Discriminator variables
  # Input weight vector is morphed to shape according to input vector
  # D_W1 = tf.Variable(xavier_init([dim*2, h_dim*2])) # Data + Hint as inputs
  D_b1 = tf.Variable(tf.zeros(shape = [h_dim*2]))

  D_W2 = tf.Variable(xavier_init([h_dim*2, h_dim*2]))
  D_b2 = tf.Variable(tf.zeros(shape = [h_dim*2]))

  D_W3 = tf.Variable(xavier_init([h_dim*2, h_dim]))
  D_b3 = tf.Variable(tf.zeros(shape = [h_dim]))

  # D_W4 = tf.Variable(xavier_init([h_dim, h_dim]))
  # D_b4 = tf.Variable(tf.zeros(shape = [h_dim])) 

  # D_W5 = tf.Variable(xavier_init([h_dim, dim]))
  # D_b5 = tf.Variable(tf.zeros(shape = [dim]))  # Multi-variate outputs
  
  # Generator variables
  # Data + Mask as inputs (Random noise is in missing components)
  # Input weight vector is morphed to shape according to input vector
  # G_W1 = tf.Variable(xavier_init([dim*2, h_dim*2]))  
  G_b1 = tf.Variable(tf.zeros(shape = [h_dim*2]))

  G_W2 = tf.Variable(xavier_init([h_dim*2, h_dim*2]))
  G_b2 = tf.Variable(tf.zeros(shape = [h_dim*2]))
  
  G_W3 = tf.Variable(xavier_init([h_dim*2, dim]))
  G_b3 = tf.Variable(tf.zeros(shape = [dim]))

  # G_W4 = tf.Variable(xavier_init([h_dim, dim]))
  # G_b4 = tf.Variable(tf.zeros(shape = [dim]))

  # G_W5 = tf.Variable(xavier_init([h_dim, dim]))
  # G_b5 = tf.Variable(tf.zeros(shape = [dim]))
 
## GAIN functions
  # Generator
  def generator(x, m):
    # Concatenate Mask and Data
    inputs = tf.concat(values = [x, m], axis = 1)
    G_W1 = tf.Variable(xavier_init([(inputs.shape[1]), h_dim*2]))
    print("inputs.shape, DW1.shape :", inputs.shape, ",", inputs.shape[1], h_dim*2)
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    # G_h3 = tf.nn.relu(tf.matmul(G_h2, G_W3) + G_b3)
    # G_h4 = tf.nn.relu(tf.matmul(G_h3, G_W4) + G_b4)
    # MinMax normalized output
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
    return G_prob, G_W1

  # Discriminator
  def discriminator(x, h):
    # Concatenate Data and Hint
    inputs = tf.concat(values = [x, h], axis = 1)
    D_W1 = tf.Variable(xavier_init([(inputs.shape[1]), h_dim*2])) # Data + Hint as inputs..
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    # D_h3 = tf.nn.relu(tf.matmul(D_h2, D_W3) + D_b3)
    # D_h4 = tf.nn.relu(tf.matmul(D_h3, D_W4) + D_b4)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    # D_prob = tf.nn.sigmoid(D_logit)
    D_prob = D_logit
    return D_prob, D_W1

  const1 = 1e-8
  ### GAN setup for reusability 
  def GAN_setup(X_func, X, M_func, M, H):
    # Generator
    print("X_func shape:", X_func.shape)
    G_sample, G_W1 = generator(X_func, M_func)
    # Combine with observed data
    Hat_X = X * M + G_sample * (1- M)
    # Discriminator
    D_prob, D_W1 = discriminator(Hat_X, H) # Output is Hat_M
    # GAIN loss
    
    D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + const1) \
                                  + (1-M) * tf.log(1. - D_prob + const1)) 
        
    G_loss_temp = -tf.reduce_mean((1-M) * tf.log(D_prob + const1))
    
    # G_loss_temp = -tf.reduce_mean(M * tf.log(1 - D_prob + const1) \
    #                              + (1-M) * tf.log(D_prob + const1))

    return G_sample, D_loss_temp, G_loss_temp, G_W1, D_W1

  const2 = 1e-10
  const3 = 1
  def MSE_calc(M_non_bin, X_non_bin, M_bin, X_bin, G_sample):
    
    G_sample_vecs = tf.unstack(G_sample, axis=1)
    G_sample_not_bin = tf.stack([ele for f, ele in enumerate(G_sample_vecs) if f not in binary_features], 1)
    MSE_loss_not_binary = tf.reduce_mean((M_non_bin * X_non_bin - M_non_bin * G_sample_not_bin)**2) / tf.reduce_mean(M_non_bin)

    if not binary_features:
      G_sample_bin = []
      MSE_loss_binary = 0
    else:
      G_sample_bin = tf.stack([ele for f, ele in enumerate(G_sample_vecs) if f in binary_features], 1)
      MSE_loss_binary = -const3 * tf.reduce_mean(M_bin * X_bin * tf.log(M_bin * G_sample_bin + const2))  
    
    MSE_loss = MSE_loss_not_binary + MSE_loss_binary
    return MSE_loss, MSE_loss_not_binary, MSE_loss_binary
  
  def G_sample_bin_corr(ori_X_bin, G_sample_bin):
    df1 = pd.DataFrame(ori_X_bin)
    df2 = pd.DataFrame(G_sample_bin)
    df = pd.concat([df1, df2], axis=1)
    df.columns = ['ori_X_bin', 'G_sample_bin']
    corr = df['ori_X_bin'].corr(df['G_sample_bin'])
    return corr

  def calc_imputed_data():
    ## Return imputed data  
    Z_mb_temp = uniform_sampler(0, 0.01, no, dim)
    M_mb_temp = data_m
    X_mb_temp = norm_data_x
    X_mb_temp = M_mb_temp * X_mb_temp + (1-M_mb_temp) * Z_mb_temp

    # Run for 1 iteration, hence no further training.
    imputed_data = sess.run([G_sample], feed_dict = {X: X_mb_temp, M: M_mb_temp})[0]
    imputed_data = data_m * norm_data_x + (1-data_m) * imputed_data

    # Renormalization
    imputed_data = renormalization(imputed_data, norm_parameters)

    if bin_category_f:
      # digitize data
      # print("Binning categorical features")
      imputed_data = digitizing(imputed_data, data_x, bins, categorical_features, binary_features)
    else:
      # Rounding
      # print("Rounding")
      imputed_data = rounding(imputed_data, data_x, categorical_features)

    return imputed_data
  
  print("categorical features", categorical_features)
  print("use_cont_f", use_cont_f)
  print("use_cat_f", use_cat_f)
  if (not categorical_features) and (not use_cont_f):
    raise Exception("Use_cont_f cannot be False when no categorical data in the database")
  else:
    M_temp = M
    X_temp = X
    H_temp = H
    M_vecs = tf.unstack(M_temp, axis=1)
    X_vecs = tf.unstack(X_temp, axis=1)
    H_vecs = tf.unstack(H_temp, axis=1)

    M_cont = tf.stack([ele for f, ele in enumerate(M_vecs) if f not in categorical_features], 1)
    X_cont = tf.stack([ele for f, ele in enumerate(X_vecs) if f not in categorical_features], 1)
    H_cont = tf.stack([ele for f, ele in enumerate(H_vecs) if f not in categorical_features], 1)

    if not categorical_features:
      M_cat = []
      X_cat = []
      H_cat = []
    else:
      M_cat = tf.stack([ele for f, ele in enumerate(M_vecs) if f in categorical_features], 1)
      X_cat = tf.stack([ele for f, ele in enumerate(X_vecs) if f in categorical_features], 1)
      H_cat = tf.stack([ele for f, ele in enumerate(H_vecs) if f in categorical_features], 1)

    if not binary_features:
      M_bin = []
      X_bin = []
      H_bin = []
    else:
      M_bin = tf.stack([ele for f, ele in enumerate(M_vecs) if f in binary_features], 1)
      X_bin = tf.stack([ele for f, ele in enumerate(X_vecs) if f in binary_features], 1)
      H_bin = tf.stack([ele for f, ele in enumerate(H_vecs) if f in binary_features], 1) 

    M_not_bin = tf.stack([ele for f, ele in enumerate(M_vecs) if f not in binary_features], 1)
    X_not_bin = tf.stack([ele for f, ele in enumerate(X_vecs) if f not in binary_features], 1)
    H_not_bin = tf.stack([ele for f, ele in enumerate(H_vecs) if f not in binary_features], 1) 

    if (not categorical_features) or (categorical_features and use_cont_f==True and use_cat_f==True):
      G_sample, D_loss_temp, G_loss_temp, G_W1, D_W1 = GAN_setup(X, X, M, M ,H)

    elif use_cont_f and not use_cat_f:
      G_sample, D_loss_temp, G_loss_temp, G_W1, D_W1 = GAN_setup(X_cont, X, M_cont, M, H)

    elif use_cat_f and not use_cont_f:
      G_sample, D_loss_temp, G_loss_temp, G_W1, D_W1 = GAN_setup(X_cat, X, M_cat, M, H)

    # print("MSE_loss_cont, MSE_loss_cat, MSE_loss :", MSE_loss_cont, MSE_loss_cat, MSE_loss)    

    # This MSE loss is for vectors which are already present: not for imputed data.
    G_sample_temp = G_sample
    G_sample_vecs = tf.unstack(G_sample_temp, 1)
    # G_sample_bin = tf.stack([ele for f, ele in enumerate(G_sample_vecs) if f in binary_features], 1) 
    # G_sample_not_bin = tf.stack([ele for f, ele in enumerate(G_sample_vecs) if f not in binary_features], 1) 

    MSE_loss, MSE_loss_not_binary, MSE_loss_binary = MSE_calc(M_not_bin, X_not_bin, M_bin, X_bin, G_sample)
    D_loss = D_loss_temp
    G_loss = G_loss_temp + alpha * MSE_loss

    # print("MSE_loss_cont, MSE_loss_cat, MSE_loss :", MSE_loss_cont, MSE_loss_cat, MSE_loss)
    # Generator Variable
    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    # Discriminator variable list
    #  theta_D = [D_W1, D_W2, D_W3, D_W4, D_b1, D_b2, D_b3, D_b4] 
    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3] 

    ## GAIN solver
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
    
    ## Iterations
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    loss_list = []
    print(" ################# TRAINING STARTS #####################")

    rmse_it = np.zeros((iterations))
    rmse_per_feature_it = np.zeros((iterations, dim))
    # print("Rmse_it shape", rmse_it.shape)
    # print("rmse_per_feature_it shape", rmse_per_feature_it.shape)

  # Start Iterations
  for it in tqdm(range(iterations)):
   # print(" --------------------iteration:", it, "-------------------------- ")
    # Sample batch
    batch_idx = sample_batch_index(no, batch_size)
    X_mb = norm_data_x[batch_idx, :]
    M_mb = data_m[batch_idx, :]

    # Sample random vectors
    Z_mb = uniform_sampler(0, 0.01, batch_size, dim)
    
    # Sample hint vectors
    H_mb_temp = binary_sampler(hint_rate, batch_size, dim)

    # print("H_mb_temp value :", H_mb_temp.dtype)
    H_mb = M_mb * H_mb_temp
    
    # Combine random vectors with observed vectors
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 

    # print("X is:") 
    # tf.print(X)
    # print("M is:", M)
    # print ("G Sample is :", G_sample)  

    _, G_loss_curr, MSE_loss_curr, MSE_loss_not_binary_curr, MSE_loss_binary_curr = \
    sess.run([G_solver, G_loss_temp, MSE_loss, MSE_loss_not_binary, MSE_loss_binary],
              feed_dict = {X: X_mb, M: M_mb, H: H_mb})

    # Sample random vectors
    noise = uniform_sampler(0, 1, batch_size, dim)

    _, D_loss_curr = sess.run([D_solver, D_loss_temp], 
                            feed_dict = {M: M_mb, X: X_mb +(noise), H: H_mb})

    ##################################################
    ######## Check RMSE after every iteration ########
    ##################################################

    if deep_analysis:
      imputed_data_it = calc_imputed_data()
      rmse_it[it], rmse_per_feature_it[it, :] = rmse_loss(ori_data_x, imputed_data_it, data_m)
      # print("MSE loss at iteration", it, ":", MSE_loss)
      # print("MSE loss current at iteration", it, ":", MSE_loss_curr)
    
      # Calculate binary correlations.
      for f in binary_features:
        # print("Ori data binary", ori_data_x[:, f])
        df = pd.DataFrame(ori_data_x[:, f] - imputed_data_it[:, f])
        pd.set_option('display.max_rows', None)
        # print("df", df)
        print("Number of bad imputation entries:", np.count_nonzero(ori_data_x[:, f]-imputed_data_it[:,f]))
        G_sample_bin_correlation.append(G_sample_bin_corr(ori_data_x[:,f], imputed_data_it[:,f]))

    loss_list.append((D_loss_curr, MSE_loss_not_binary_curr, MSE_loss_binary_curr, MSE_loss_curr, G_loss_curr))

  ## Return imputed data
  imputed_data = calc_imputed_data()
  
  import matplotlib.pyplot as plt
  x = np.arange(len(G_sample_bin_correlation))
  fig = plt.figure()
  plt.plot(x, G_sample_bin_correlation)
  
  if deep_analysis:
    return imputed_data, loss_list, rmse_it, rmse_per_feature_it
  else:
    return imputed_data, loss_list
