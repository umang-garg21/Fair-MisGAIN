import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
from collections import namedtuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from load_adult import load_adult

def main():
    smaller = False
    scalar = True
    drop_p = 0.1

    a,b = load_adult(smaller, scalar, drop_p)  # currently drop_p is not used

    print("Modified Adult train data", a)
    print("/n Modified Adult test data", b)

if __name__ == "__main__":
    main()
