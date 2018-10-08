# purchasing feature analysis

import pandas as pd
import numpy as np
from numpy import array
import psycopg2
import math
import traceback
import matplotlib.pyplot as plt
import lightgbm as lgb
from collections import defaultdict

from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, classification_report, roc_curve, auc  

#import seaborn as sns

# output path
input_path = "C:/Users/yawen/Google Drive/spatial temporal data analytics/urban computing/PurchasingBehavior/feature_analysis_output/"
output_path = "C:/Users/yawen/Google Drive/spatial temporal data analytics/urban computing/PurchasingBehavior/feature_analysis_output/"

# connect to database and get cursor
try:
    conn = psycopg2.connect(database = 'tmall', user = 'postgres', host = '128.138.157.86', port = '5432', password = '1234')

except psycopg2.Error as e:
    print("I am unable to connect to the database")
    print(e)
    print(e.pgcode)
    print(e.pgerror)
    print(traceback.format_exc())

cur = conn.cursor()

def features():
    """
    get all features
    """
    cur.execute("select * from train_ui_up_mp_um_double_pca_sim_slope_fillna")
    train_ui_up_mp_um_double_pca_sim_slope_fillna_features = pd.DataFrame(cur.fetchall(), columns = [i[0] for i in cur.description])

    return train_ui_up_mp_um_double_pca_sim_slope_fillna_features

    # train_data_user_merchant_profile_features_stat = train_data_user_merchant_profile_features.groupby('label').describe()
    # train_data_user_merchant_profile_features_stat.to_csv(output_path + "train_data_user_merchant_profile_features_stat.csv")

    # check missing value perc
    # print(user_info.isnull().sum())
    # print(user_label[user_label['label'] == 1])

    # plot the histgram for columns
    # user_label.hist(column = 'label', bins = 2, grid = False)
    # plt.xticks(np.arange(-2, 2, 1))
    # plt.show()

def classification(train_ui_up_mp_um_double_pca_sim_slope_fillna_features):

    # ignore NaN, if any value is NaN
    # train_data_user_merchant_profile_features = train_data_user_merchant_profile_features.dropna(how = 'any')
    x = train_ui_up_mp_um_double_pca_sim_slope_fillna_features.filter(regex = 'mp')
    x = x.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    y = train_ui_up_mp_um_double_pca_sim_slope_fillna_features['label']

    train_size = [50000, 100000, 150000, 200000]
    test_size = [10000, 20000, 30000, 50000]

    for i in range(3, 4):
        train_num = train_size[i]
        test_num = test_size[i]

        x_train_o = x.iloc[:train_num]
        y_train_o = y.iloc[:train_num]

        # undersampling
        repeat_indices = y_train_o[y_train_o == 1].index
        print(len(repeat_indices))
        repeat_random_indices = np.random.choice(repeat_indices, int(train_size[i] * 0.05))
        print(len(repeat_random_indices))

        not_repeat_indices = y_train_o[y_train_o == 0].index
        not_repeat_random_indices = np.random.choice(not_repeat_indices, int(train_size[1] * 0.20))

        x_train = x_train_o.iloc[np.concatenate((repeat_random_indices, not_repeat_random_indices))]
        print(x_train.shape)
        y_train = y_train_o.iloc[np.concatenate((repeat_random_indices, not_repeat_random_indices))]
        print(y_train.shape)

        x_test = x.iloc[-test_num:]
        y_test = y.iloc[-test_num:]

        ### scale the data first
        scaler = preprocessing.StandardScaler().fit(x_train)
        scaler.transform(x_train)
        scaler.transform(x_test)

### GBM models-----------------------------------------------------------------------------------------    
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

        # specify your configurations as a dict
        params = {
            'task': 'predict',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l2', 'auc'},
            'num_leaves': 91,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }

       # print('Start training...')
        # train
        gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)

        #print('Save model...')
        # save model to file
        gbm.save_model('model.txt')

        

        #print('Start predicting...')
        # predict
        pre_GBM = gbm.predict(x_test)
        
        # feature importances
        #print('Feature importances:', list(gbm.feature_importances_))
        # other scikit-learn modules
        estimator = lgb.LGBMRegressor(num_leaves=31)
        param_grid = {
            #'learning_rate': [0.001,0.01,0.05,0.1,0.15,1]
            #'n_estimators': [20, 40,60,80,100,120,140,160,180,200],
            #'num_leaves': [5,10,15,20,30,40,50,60,70,80,100]
        }
        gbm = GridSearchCV(estimator, param_grid)
        gbm.fit(x_train, y_train)
        print('Best parameters found by grid search are:', gbm.best_params_)

       
        # confusion matrix
        '''confusion = confusion_matrix(y_test, pre_GBM)
        print("\t" + "\t".join(str(x) for x in range(0, 2)))
        print("".join(["-"] * 50))
        for ii in range(0, 2):
            jj = ii
            print("%i:\t" % jj + "\t".join(str(confusion[ii][x]) for x in range(0, 2)))
        '''
        #print(pre)
        #print(f1_score(y_test, predictions))
        #print(precision_score(y_test, predictions))
        #print(recall_score(y_test, predictions))
        # print(roc_auc_score(y_test, predictions))
        # print(classification_report(y_test, predictions))
        fpr, tpr, thresholds = roc_curve(y_test, pre_GBM)  
        roc_auc = auc(fpr, tpr)  
        print(roc_auc)

    print('Finish!')

if __name__ == "__main__":

    train_ui_up_mp_um_double_pca_sim_slope_fillna_features = features()
    classification(train_ui_up_mp_um_double_pca_sim_slope_fillna_features)