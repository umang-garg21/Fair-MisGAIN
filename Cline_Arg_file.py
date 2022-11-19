hyperparameters = {"--file":'main.py', "--data_name":'adult', "--miss_rate" :0.1, "--batch_size": 128, "--hint_rate": 0.9, "-alpha": 100, "--iterations": 100, "--runs": 5, "--drop_f": 13, "--imputer": "Gain"}

model = (n_neighbors=5, algorithm="ball_tree")
model = ( **hyperparameters)