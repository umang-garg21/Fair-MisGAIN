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

from utils import normalization, renormalization, rounding
from utils import xavier_init
from utils import binary_sampler, uniform_sampler, sample_batch_index
from utils import rmse_loss


def gain (ori_data_x, data_x, gain_parameters, schedule, categorical_features=[], deep_analysis=False):
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
  
  # Other parameters
  no, dim = data_x.shape
  
  # Hidden state dimensions
  h_dim = int(dim)
  
  # Normalization
  norm_data, norm_parameters = normalization(data_x)
  norm_data_x = np.nan_to_num(norm_data, 0)
  
  ## GAIN architecture
  # Input placeholders
  # Data vector
  X = tf.placeholder(tf.float32, shape = [None, dim])
  # Mask vector
  M = tf.placeholder(tf.float32, shape = [None, dim])
  # Hint vector
  H = tf.placeholder(tf.float32, shape = [None, dim])
  
  # Discriminator variables
  D_W1 = tf.Variable(xavier_init([dim*2, h_dim*2])) # Data + Hint as inputs
  D_b1 = tf.Variable(tf.zeros(shape = [h_dim*2]))

  D_W2 = tf.Variable(xavier_init([h_dim*2, h_dim*2]))
  D_b2 = tf.Variable(tf.zeros(shape = [h_dim*2]))   

  D_W3 = tf.Variable(xavier_init([h_dim*2, h_dim]))
  D_b3 = tf.Variable(tf.zeros(shape = [h_dim]))    

 # D_W4 = tf.Variable(xavier_init([h_dim, h_dim]))
 # D_b4 = tf.Variable(tf.zeros(shape = [h_dim])) 

#  D_W5 = tf.Variable(xavier_init([h_dim, dim]))
#  D_b5 = tf.Variable(tf.zeros(shape = [dim]))  # Multi-variate outputs
  
#  theta_D = [D_W1, D_W2, D_W3, D_W4, D_b1, D_b2, D_b3, D_b4] 
  theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3] 

  #Generator variables
  # Data + Mask as inputs (Random noise is in missing components)
  G_W1 = tf.Variable(xavier_init([dim*2, h_dim*2]))  
  G_b1 = tf.Variable(tf.zeros(shape = [h_dim*2]))

  G_W2 = tf.Variable(xavier_init([h_dim*2, h_dim*2]))
  G_b2 = tf.Variable(tf.zeros(shape = [h_dim*2]))
  
  G_W3 = tf.Variable(xavier_init([h_dim*2, dim]))
  G_b3 = tf.Variable(tf.zeros(shape = [dim]))

#  G_W4 = tf.Variable(xavier_init([h_dim, dim]))
#  G_b4 = tf.Variable(tf.zeros(shape = [dim]))

#  G_W5 = tf.Variable(xavier_init([h_dim, dim]))
#  G_b5 = tf.Variable(tf.zeros(shape = [dim]))
  
  theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]
 
  ## GAIN functions
  # Generator
  def generator(x,m):
    # Concatenate Mask and Data
    inputs = tf.concat(values = [x, m], axis = 1)
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
  #  G_h3 = tf.nn.relu(tf.matmul(G_h2, G_W3) + G_b3)
  #  G_h4 = tf.nn.relu(tf.matmul(G_h3, G_W4) + G_b4)
    # MinMax normalized output
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
    return G_prob
  
  # Discriminator
  def discriminator(x, h):
    # Concatenate Data and Hint
    inputs = tf.concat(values = [x, h], axis = 1) 
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)  
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
 #  D_h3 = tf.nn.relu(tf.matmul(D_h2, D_W3) + D_b3)
 #  D_h4 = tf.nn.relu(tf.matmul(D_h3, D_W4) + D_b4)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob
  
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

    # Rounding
    imputed_data = rounding(imputed_data, data_x, categorical_features)
    
    return imputed_data

  ## GAIN structure
  # Generator
  G_sample = generator(X, M)
  
  # Combine with observed data
  Hat_X = X * M + G_sample * (1-M)   
  
  # Discriminator
  D_prob = discriminator(Hat_X, H) # Output is Hat_M
  
  ## GAIN loss
  D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) \
                                + (1-M) * tf.log(1. - D_prob + 1e-8)) 
  
  G_loss_temp = -tf.reduce_mean((1-M) * tf.log(D_prob + 1e-8))

  # This MSE loss is for vector which are already present: not for imputed data.
  # MSE_loss different for continious and binary features.
  if not categorical_features:
    MSE_loss = tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)
    D_loss = D_loss_temp
    G_loss = G_loss_temp + alpha * MSE_loss
  else:
    M_temp = M
    X_temp = X
    Gsample_temp = G_sample
    M_vecs = tf.unstack(M_temp, axis=1)
    X_vecs = tf.unstack(X_temp, axis=1)
    Gsample_vecs = tf.unstack(Gsample_temp, axis=1)

    M_cont = tf.stack([ele for f, ele in enumerate(M_vecs) if f not in categorical_features], 1)
    M_cat = tf.stack([ele for f, ele in enumerate(M_vecs) if f in categorical_features], 1)

    X_cont = tf.stack([ele for f, ele in enumerate(X_vecs) if f not in categorical_features], 1)
    X_cat = tf.stack([ele for f, ele in enumerate(X_vecs) if f in categorical_features], 1)

    Gsample_cont = tf.stack([ele for f, ele in enumerate(Gsample_vecs) if f not in categorical_features], 1)
    Gsample_cat = tf.stack([ele for f, ele in enumerate(Gsample_vecs) if f in categorical_features], 1)

    MSE_loss_cont = tf.reduce_mean((M_cont * X_cont - M_cont * Gsample_cont)**2) / tf.reduce_mean(M_cont)
    MSE_loss_cat = -tf.reduce_mean(M_cat * X_cat * tf.log(M_cat * Gsample_cat + 1e-8)) / tf.reduce_mean(M_cat)

    MSE_loss = 0
    use_categorical_data = False
    use_cont_data = True
    if use_cont_data:
      MSE_loss = MSE_loss + MSE_loss_cont
    if use_categorical_data:
      MSE_loss = MSE_loss + MSE_loss_cat

    # print("MSE_loss_cont, MSE_loss_cat, MSE_loss :", MSE_loss_cont, MSE_loss_cat, MSE_loss)

    D_loss = D_loss_temp
    G_loss = G_loss_temp + alpha * MSE_loss

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
      
    _, D_loss_curr = sess.run([D_solver, D_loss_temp], 
                              feed_dict = {M: M_mb, X: X_mb, H: H_mb})

    # print("X is:") 
    # tf.print(X)
    # print("M is:", M)
    # print ("G Sample is :", G_sample)  

    _, G_loss_curr, MSE_loss_curr, MSE_loss_cont_curr, MSE_loss_cat_curr = \
    sess.run([G_solver, G_loss_temp, MSE_loss, MSE_loss_cont, MSE_loss_cat],
              feed_dict = {X: X_mb, M: M_mb, H: H_mb})

    #################################
    # Check RMSE after every iteration 
    #################################

    if deep_analysis:
      imputed_data_it = calc_imputed_data()
      rmse_it[it], rmse_per_feature_it[it, :] = rmse_loss(ori_data_x, imputed_data_it, data_m)

    # print("MSE loss at iteration", it, ":", MSE_loss)
    # print("MSE loss current at iteration", it, ":", MSE_loss_curr)

    loss_list.append((D_loss_curr, MSE_loss_cont_curr, MSE_loss_cat_curr, MSE_loss_curr, G_loss_curr))
    
  ## Return imputed data      
  imputed_data = calc_imputed_data()
  
  if deep_analysis:
    return imputed_data, loss_list, rmse_it, rmse_per_feature_it
  else:
    return imputed_data, loss_list
