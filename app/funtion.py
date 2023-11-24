import pandas as pd
import numpy as np
import csv

df = pd.read_csv("static\\Trainingdata.csv")

COUNTRY = 'country'
YEAR = 'year'

# Features 
GDP = 'real_gdp_cap'
INSTABILITY = 'instability_idx'
POP = 'population'
GENDER = 'gender_inequality_index'
RAINFALL = 'tot_rainfall_yr'
TEMP = 'ave_temp'
CONFLICT = 'conflict_death_rate'
CPI = 'food_CPI_inflation'
CORRUPTION = 'corruption_percention_idx'
LPI_TIME = 'lpi_timeliness'
LPI_Q = 'lpi_logs_competence_quality'
AGRI_PROD = 'domestic_agricultural_production_primary_crops'
AGRI_EX = 'domestic_agricultural_export_primary_crops'
AGRI_IM = 'domestic_agricultural_import_primary_crops'

# Targets
TARGET_AFFORD = 'gfsi_affordability_idx'
TARGET_AVAIL = 'gfsi_availability_idx'

countries = ['DR CONGO', 'ANGOLA', 'TANZANIA', 'ZAMBIA', 'UGANDA', 'MOZAMBIQUE', 'BOTSWANA', 'RWANDA', 'MALAWI', 'BURUNDI']
features = [GDP, INSTABILITY, POP, GENDER, RAINFALL, TEMP, CONFLICT, CPI, CORRUPTION, LPI_TIME, LPI_Q, AGRI_PROD, AGRI_EX, AGRI_IM]
targets = [TARGET_AFFORD, TARGET_AVAIL]
targeted_features = {
    TARGET_AFFORD:[GDP, INSTABILITY, POP, GENDER, RAINFALL, TEMP, CONFLICT, CPI, LPI_TIME, LPI_Q, AGRI_PROD, AGRI_EX, AGRI_IM],
    TARGET_AVAIL: [GDP, INSTABILITY, POP, GENDER, RAINFALL, TEMP, CORRUPTION, LPI_TIME, LPI_Q, AGRI_EX, AGRI_IM]
}
feature_linearisations_by_target = {
    TARGET_AFFORD: {
        INSTABILITY: lambda a: a**3,
        RAINFALL   : lambda a: np.log(a),
        CPI        : lambda a: np.exp(-0.022*a),
        LPI_TIME   : lambda a: np.log(a),
        LPI_Q      : lambda a: np.log(a)
    },
    TARGET_AVAIL: {
        CORRUPTION: lambda a: np.log(a),
        AGRI_EX   : lambda a: np.log(a)
    }
}
FACTOR = 0.3 # determined proportion of test data

def split_test_train(df_in:pd.DataFrame, countries:list[str], seed:int=100, factor:int=FACTOR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns training and test set
    """
    # select data, fixed ratio of train:test per country
    df_train = pd.DataFrame(columns=df_in.columns)
    df_test  = pd.DataFrame(columns=df_in.columns)
    np.random.seed(seed)

    for country in countries:

        df_temp = df_in.loc[df[COUNTRY] == country, :]
        idxs    = df_temp.index

        test_idx  = sorted(np.random.choice(idxs, int(df_temp.shape[0]*factor), replace=False))
        train_idx = [idx for idx in idxs if not idx in test_idx]

        df_test  = pd.concat([df_test,  df_in.loc[test_idx,  :]])
        df_train = pd.concat([df_train, df_in.loc[train_idx, :]])

    return df_train, df_test

def split_feature_target(df_train, df_test, features:list[str], target:list[str]):
    """
    Annotate functions later 
    """
    df_features_train = df_train[features]
    df_features_test  = df_test[features]
    df_target_train   = df_train[target]
    df_target_test    = df_test[target]
    return df_features_train, df_features_test, df_target_train, df_target_test

def normalize_z(dfin, columns_means=None, columns_stds=None) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if columns_means is None:
            columns_means = dfin.mean(axis = 0)
    if columns_stds is None:
        columns_stds = dfin.std(axis=0)
    dfout = (dfin - np.array(columns_means))/np.array(columns_stds)
    return dfout, columns_means, columns_stds

def get_features_targets(df, feature_names: list[str], target_names: list[str]):
    df_feature = df[feature_names] 
    df_target = df[target_names] 
    return df_feature, df_target

def prepare_feature(df_feature) -> np.ndarray:
    """Accepts dataframe and ndarray"""
    if isinstance(df_feature, pd.DataFrame):
        np_feature = df_feature.to_numpy()
        cols = df_feature.shape[1]
    elif isinstance(df_feature, pd.Series):
        np_feature = df_feature.to_numpy()
        cols = 1
    else:
        np_feature = df_feature
        cols = df_feature.shape[1]

    feature = np_feature.reshape(-1,cols)
    columns_ones = np.ones((feature.shape[0], 1))
    X = np.concatenate((columns_ones, feature), axis=1)
    return X

def prepare_target(df_target: pd.DataFrame) -> np.ndarray:
    if isinstance(df_target, pd.DataFrame):
        np_target = df_target.to_numpy()
    else:
        np_target = df_target
    cols = df_target.shape[1]
    # check keyword argument call for np reshape 
    target = np_target.reshape(-1,cols)
    return target

def predict_linreg(df_feature: pd.DataFrame, beta, means=None, stds=None):
    """Returns predicted y hat values"""
    # norm data is pd.Dataframe
    norm_data,_,_ = normalize_z(df_feature, columns_means=means, columns_stds=stds)
    
    X = prepare_feature(norm_data)
    return calc_linreg(X, beta)

def calc_linreg(X, beta) -> np.ndarray:
    return np.matmul(X, beta)

def compute_cost_linreg(X, y, beta):
    m = X.shape[0]
    error = calc_linreg(X, beta)-y
    J = (np.matmul(error.T, error)/(2*m))[0][0]
    return J

def gradient_descent_linreg(X:pd.DataFrame, y, beta, alpha, num_iters):
    m = X.shape[0] # number of rows 
    J_storage = np.zeros((num_iters,1))
    for n in range(num_iters):
        yhat: np.ndarray = calc_linreg(X,beta)
        error = yhat - y
        deriv = np.matmul(X.T,error)/m # shape is (2x1)
        beta = beta - alpha * deriv # beta (2 x 1) vector
        J_storage[n] = compute_cost_linreg(X, y, beta)
    return beta, J_storage
  
def r2_score(y: np.ndarray, ypred: np.ndarray):
    ymean: np.float64 = np.mean(y)
    error_mean = y - ymean 
    # sstot is [[number]]
    sstot = np.matmul(error_mean.T, error_mean)
    error = y - ypred
    # ssres is [[number]]
    ssres = np.matmul(error.T, error)
    r2 = 1 - (ssres / sstot)
    return r2[0][0]

def adjusted_r2(y: np.ndarray, ypred: np.ndarray, n:int = int(df.shape[0]*(1-FACTOR)), p:int = len(features)):
    r2 = r2_score(y, ypred)
    return 1-(((1-r2)*(n-1))/(n-p-1))

def mean_squared_error(target, pred):
    return np.sum((target - pred)**2)/target.shape[0]

def linearise_features(df:pd.DataFrame, func_dict:dict[str,callable]):
    df_out = df.copy()
    for feature, func in func_dict.items():
        df_out[feature] = df[feature].apply(func)
    return df_out

def prediction_model(df: pd.DataFrame, countries: list[str], targeted_features: dict[str, list[str]], linearise_dict:dict[str, callable]):
    """
    Multiple linear regression model 
    """
    df_train, df_test = split_test_train(df, countries, seed=0, factor = FACTOR)

    betas        = {}
    preds        = {}
    feature_tests = {}
    target_tests = {}
    means = {}
    stds = {}
    
    for t in targeted_features:
        features = targeted_features[t]
        df_features_train, df_features_test, df_target_train, df_target_test = split_feature_target(df_train, df_test, features, [t])

        # Normalize the features using z normalization
        df_features_train = linearise_features(df_features_train, linearise_dict[t])
        df_features_train_z,mean,std = normalize_z(df_features_train)
        means[t] = mean
        stds[t] = std

        # Change the features and the target to numpy array using the prepare functions
        X = prepare_feature(df_features_train_z)
        target = prepare_target(df_target_train)

        iterations = 1500
        alpha = 0.01
        beta = np.zeros((len(features)+1,1))

        # Call the gradient_descent function
        beta, _ = gradient_descent_linreg(X,target,beta,alpha,iterations)
        betas[t] = beta

        # call the predict() method
        df_features_test = linearise_features(df_features_test, linearise_dict[t])
        pred = predict_linreg(df_features_test, beta, mean, std)
        preds[t] = pred
        feature_tests[t]  = df_features_test
        target_tests[t]   = prepare_target(df_target_test)

    return betas, preds, feature_tests, target_tests, means, stds

#run the predict function and get affordability and availability as dictionary
def run_idx(data_file_path):
    betas, preds, features_tests, target_tests, means, stds = prediction_model(df, countries, targeted_features, feature_linearisations_by_target)
    df_this_year = pd.read_csv(data_file_path)

    next_year_predictions = {}
    pretable = {}

    for t in targets:
        df_features_prediction,_ = get_features_targets(df_this_year, targeted_features[t], t) 
        df_features_prediction = linearise_features(df_features_prediction, feature_linearisations_by_target[t])
        prediction_countries = [x[0] for x in df_this_year[[COUNTRY]].values.tolist()] 
        next_year_predictions[t] = predict_linreg(df_features_prediction, betas[t], means[t], stds[t]) 

    affordability = next_year_predictions['gfsi_affordability_idx']
    availability = next_year_predictions['gfsi_availability_idx']
    
    pretable[0] = prediction_countries
    pretable[1] = [x[0] for x in affordability.tolist()]
    pretable[2] = [x[0] for x in availability.tolist()]
    return pretable

#function to convert dictionary to csv aand store in folder
def convertcsv(inx):
    dic = {}
    test = []
    field_names = ['Country', 'Afford_idx', 'Ava_idx'] 
    for idx in range(len(inx[0])):
        dic["Country"] = inx[0][idx]
        dic["Afford_idx"] = inx[1][idx]
        dic["Ava_idx"] = inx[2][idx]
        test.append(dic.copy())

    with open("static\\csv\\Data.csv", 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(test)
    return 





    

