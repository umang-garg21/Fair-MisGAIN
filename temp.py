      
    ## GAIN structure
    # MSE_loss different for continious and binary features.
  if not categorical_features and not use_cont_f:
    raise Exception("Use_cont_f cannot be False when no categorical data in the database")
  else:
    M_temp = M
    X_temp = X
    H_temp = H
    M_vecs = tf.unstack(M_temp, axis=1)
    X_vecs = tf.unstack(X_temp, axis=1)
    H_vecs = tf.unstack(H_temp, axis=1)

    M_cont = tf.stack([ele for f, ele in enumerate(M_vecs) if f not in categorical_features], 1)
    M_cat = tf.stack([ele for f, ele in enumerate(M_vecs) if f in categorical_features], 1)

    X_cont = tf.stack([ele for f, ele in enumerate(X_vecs) if f not in categorical_features], 1)
    X_cat = tf.stack([ele for f, ele in enumerate(X_vecs) if f in categorical_features], 1)

    H_cont = tf.stack([ele for f, ele in enumerate(H_vecs) if f not in categorical_features], 1)
    H_cat = tf.stack([ele for f, ele in enumerate(H_vecs) if f in categorical_features], 1)

    if ((not categorical_features) or (categorical_features==True and use_cont_f==True and use_cat_f==True)):
      G_sample, D_loss_temp, G_loss_temp = GAN_setup(X, M ,H)
      # print(" -----------------------------DEBUG-----------------------------------")

    elif use_cont_f and not use_cat_f:
      Gsample, D_loss_temp, G_loss_temp = GAN_setup(X_cont, M_cont, H_cont)

    elif use_cat_f and not use_cont_f:
      Gsample, D_loss_temp, G_loss_temp = GAN_setup(X_cat, M_cat, H_cat)

    # print("MSE_loss_cont, MSE_loss_cat, MSE_loss :", MSE_loss_cont, MSE_loss_cat, MSE_loss)

    # This MSE loss is for vectors which are already present: not for imputed data.
    MSE_loss = MSE_calc(M_cont, X_cont, M_cat, X_cat, Gsample)
    D_loss = D_loss_temp
    G_loss = G_loss_temp + alpha * MSE_loss
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