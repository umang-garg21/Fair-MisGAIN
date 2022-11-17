# Codebase for "Generative Adversarial Imputation Networks (GAIN)"

Authors: Umang Garg

This directory contains implementations of GAIN framework for imputation
using advanced datasets like UCI Adult, COMPAS.

To run the pipeline for training and evaluation on GAIN framwork, simply run 
python -m main.py.

Note that any model architecture can be used as the generator and 
discriminator model such as multi-layer perceptrons or CNNs. 

### Command inputs:

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
  - rmse_lst: RMSE data list for all runs

### Example command

```shell
python main.py --data_name adult  --miss_rate 0.1 --batch_size 128 --hint_rate 1  --alpha 10 --iterations 1000 --drop_f 0 1 5 10 13 --deep_analysis True --runs 10
```

### Outputs

-   imputed_data_x: imputed data
-   rmse: Root Mean Squared Error
