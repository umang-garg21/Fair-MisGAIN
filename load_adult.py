import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
from collections import namedtuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_adult(smaller=False, scaler=True, drop_p = 0.1):
    '''
    :param smaller: selecting this flag it is possible to generate a smaller version of the training and test sets.
    :param scaler: if True it applies a StandardScaler() (from sklearn.preprocessing) to the data.
    :return: train and test data.

    Features of the Adult dataset:
    0. age: continuous.
    1. workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    2. fnlwgt: continuous.
    3. education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th,
    Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    4. education-num: continuous.
    5. marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed,
    Married-spouse-absent, Married-AF-spouse.
    6. occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty,
    Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv,
    Protective-serv, Armed-Forces.
    7. relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    8. race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    9. sex: Female, Male.
    10. capital-gain: continuous.
    11. capital-loss: continuous.
    12. hours-per-week: continuous.
    13. native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc),
    India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico,
    Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala,
    Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
    (14. label: <=50K, >50K)
    '''
    pwd = '../../datasets/adult/'
    data = pd.read_csv(
        pwd+'adult.data',
        names=[
            "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
            "hours-per-week", "native-country", "income"]
            )
    len_train = len(data.values[:, -1])

    print("len_train", len_train)

    data_test = pd.read_csv(
        pwd+'adult.test',
        names=[
            "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
            "hours-per-week", "native-country", "income"]
    )
    data = pd.concat([data, data_test])
    print("data after concatenation", data)
 
    # Considering the relative low portion of missing data, we discard rows with missing data

    domanda = data["workclass"][4].values[1]  # It is a '?': to get the representation of the missing data

    len_train = len(data.values[:, -1])

    # print("domanda", domanda) 
    data = data[data["workclass"] != domanda]
    data = data[data["occupation"] != domanda]
    data = data[data["native-country"] != domanda]

    len_train = len(data.values[:, -1])
    # print("len_train after augmentation", len_train)

    data, data_test = data.drop(columns=['fnlwgt']), data_test.drop(columns=['fnlwgt'])

    # Here we apply discretisation on column marital_status
    #    data.replace(['Divorced', 'Married-AF-spouse',
    #              'Married-civ-spouse', 'Married-spouse-absent',
    #              'Never-married', 'Separated', 'Widowed'],
    #             ['not married', 'married', 'married', 'married',
    #              'not married', 'not married', 'not married'], inplace=True)
    
    # categorical fields
    category_col = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                    'relationship', 'sex', 'native-country', 'income']

    for f_num, col in enumerate(list(data.columns.values)):
        if col in category_col:
        # print("Column datatype:", data[col].dtypes)
        # print("Column is :", col)
        # print("Column data is: ", data[col])
        # x = data[col].unique()
        # print("x:", x)
            b, c = np.unique(data[col], return_inverse=True)
            data[col] = c

    """ For introducing missing values
    
    ix = [(row, col) for row in range(data.shape[0]) for col in range(data.shape[1])]
    for row, col in random.sample(ix, int(round(drop_p*len(ix)))):
        data.iat[row, col] = np.nan

    ix = [(row, col) for row in range(data_test.shape[0]) for col in range(data_test.shape[1])]
    for row, col in random.sample(ix, int(round(drop_p*len(ix)))):
        data.iat[row, col] = np.nan
    """

    datamat = data.values
    target = np.array([-1.0 if val == 0 else 1.0 for val in np.array(datamat)[:, -1]])
    datamat = datamat[:, :-1]

    if scaler:
        scaler = MinMaxScaler()
        scaler.fit(datamat)
        data.iloc[:, :-1] = scaler.fit_transform(datamat)
#         data.iloc[:, -1] = target


    return data.iloc[:len_train], data.iloc[len_train:]