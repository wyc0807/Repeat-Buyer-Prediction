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
    
    x = train_ui_up_mp_um_double_pca_sim_slope_fillna_features.drop(['user_id','merchant_id','label'],axis=1)#.filter( items = ['pca1','pca2','pca3','pca4','pca5','pca6','pca7','pca8','pca9','pca10','sim','up_cart_slope','up_purchase_slope','up_favorite_slope','mp_click_slope','mp_cart_slope','mp_purchase_slope','mp_favorite_slope','um_click_slope','um_cart_slope','um_purchase_slope','um_favorite_slope'])#(regex = '')
    x = x.apply(lambda x: 0 if((x.max()==0) and (x.min()==0)) else (x - x.min()) / (x.max() - x.min()))
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



        ###lightgbm-----------------------------------------------------------------------
        # create dataset for lightgbm
        print('Start GBM')
        x_train_GBM = x_train
        y_train_GBM = y_train
        x_test_GBM = x_test
        y_test_GBM = y_test
        lgb_train = lgb.Dataset(x_train_GBM, y_train_GBM)
        lgb_eval = lgb.Dataset(x_test_GBM, y_test_GBM, reference=lgb_train)

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
        pre_GBM = gbm.predict(x_test_GBM)
        
        # feature importances
        #print('Feature importances:', list(gbm.feature_importances_))
        # other scikit-learn modules
        '''estimator = lgb.LGBMRegressor(num_leaves=31)
        param_grid = {
            'learning_rate': [0.01, 0.1, 1],
            'n_estimators': [20, 40]
        }
        gbm = GridSearchCV(estimator, param_grid)
        gbm.fit(x_train, y_train)
        print('Best parameters found by grid search are:', gbm.best_params_)
        '''

        ###lightgbm rf
        ###lightgbm-----------------------------------------------------------------------
        print('Start lightgbm_RF')
        # specify your configurations as a dict
        params = {
            'task': 'predict',
            'boosting_type': 'rf',
            'objective': 'regression',
            'metric': {'l2', 'auc'},
            'num_leaves': 91,
            'learning_rate': 0.15,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }

        #print('Start training...')
        # train
        gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)

        #print('Save model...')
        # save model to file
        gbm.save_model('model_rf.txt')

        
        #print('Start predicting...')
        # predict
        pre_lightRF = gbm.predict(x_test_GBM)

        ###LR model-----------------------------------------------------------------------------------------    
        print('Start LR')
        model = LogisticRegression(class_weight = 'balanced')
        model_LR = LogisticRegression(class_weight = 'balanced')
        x_train_LR = x_train
        y_train_LR = y_train
        x_test_LR = x_test
        model_LR.fit(x_train_LR, y_train_LR)
        pre_LR = model_LR.predict_proba(x_test_LR)
        fpr_LR, tpr_LR, thresholds_LR = roc_curve(y_test, pre_LR[:,1])  
        roc_auc_LR = auc(fpr_LR, tpr_LR)  
        print(roc_auc_LR)

        ###grid search for LR
        #param_test1 = { 'min_samples_split':[10],'min_samples_leaf':[10]}
        #gsearch1 = GridSearchCV(estimator = RandomForestClassifier(n_estimators = 70,
                        #max_features='sqrt' ,random_state=10,class_weight = 'balanced',max_depth =22), 
                       #param_grid = param_test1, scoring='roc_auc',cv=5)
        #gsearch1.fit(x_train, y_train)
        #print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)

        ###SVM model------------------------------------------------------------------------------------------
        ###SVM returns only label
        # model = svm.SVC(kernel = 'linear', decision_function_shape = 'ovr')
        print('Start SVM')
        model_SVM = svm.SVC(C= 10.0, gamma=9.0, kernel = 'rbf', decision_function_shape = 'ovr', class_weight = 'balanced', probability = True)
        x_train_SVM = x_train
        y_train_SVM = y_train
        x_test_SVM = x_test
        model_SVM.fit(x_train_SVM, y_train_SVM)
        pre_SVM = model_SVM.predict_proba(x_test_SVM)
        fpr_SVM, tpr_SVM, thresholds_SVM = roc_curve(y_test, pre_SVM[:,1])  
        roc_auc_SVM = auc(fpr_SVM, tpr_SVM)  
        print(roc_auc_SVM)
        ###grid search for SVM
        #param_test1 = { 'min_samples_split':[10],'min_samples_leaf':[10]}
        #gsearch1 = GridSearchCV(estimator = RandomForestClassifier(n_estimators = 70,
                        #max_features='sqrt' ,random_state=10,class_weight = 'balanced',max_depth =22), 
                       #param_grid = param_test1, scoring='roc_auc',cv=5)
        #gsearch1.fit(x_train, y_train)
        #print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)

        ### RF model--------------------------------------------------------------------------------------------
        ### RandomForest could return the proba
        # model = RandomForestClassifier(n_estimators=70, min_samples_split=10, min_samples_leaf=10,max_depth =22,max_features='sqrt' ,random_state=10,class_weight = 'balanced')
        print('Start RF')
        model_RF = RandomForestClassifier(n_estimators=70, min_samples_split=10, min_samples_leaf=10,max_depth =22,max_features='sqrt' ,random_state=10,class_weight = 'balanced')
        x_train_RF = x_train
        y_train_RF = y_train
        x_test_RF = x_test
        model_RF.fit(x_train_RF,y_train_RF)
        pre_RF = model_RF.predict_proba(x_test_RF)
        fpr_RF, tpr_RF, thresholds_RF = roc_curve(y_test, pre_RF[:,1])  
        roc_auc_RF = auc(fpr_RF, tpr_RF)  
        print(roc_auc_RF)
        
        ### grid search for RF
        #param_test1 = { 'min_samples_split':[10],'min_samples_leaf':[10]}
        #gsearch1 = GridSearchCV(estimator = RandomForestClassifier(n_estimators = 70,
                        #max_features='sqrt' ,random_state=10,class_weight = 'balanced',max_depth =22), 
                       #param_grid = param_test1, scoring='roc_auc',cv=5)
        #gsearch1.fit(x_train, y_train)
        #print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
        
        ###Ada model-----------------------------------------------------------------------------------------------
        ###adaboost could return the proba
        # model = AdaBoostClassifier( base_estimator = RandomForestClassifier(n_estimators=70, min_samples_split=10, min_samples_leaf=10,max_depth =22,max_features='sqrt' ,random_state=10,class_weight = 'balanced'),

                         #algorithm="SAMME",
                         #n_estimators=50, learning_rate=0.1)
        print('Start Ada')
        model_Ada = AdaBoostClassifier( base_estimator = RandomForestClassifier(n_estimators=70, min_samples_split=10, min_samples_leaf=10,max_depth =22,max_features='sqrt' ,random_state=10,class_weight = 'balanced'),

                        algorithm="SAMME",
                        n_estimators=50, learning_rate=0.1)
        x_train_Ada = x_train
        y_train_Ada = y_train
        x_test_Ada = x_test
        model_Ada.fit(x_train_Ada,y_train_Ada)
        pre_Ada = model_Ada.predict_proba(x_test_Ada)
        fpr_Ada, tpr_Ada, thresholds_Ada = roc_curve(y_test, pre_Ada[:,1])  
        roc_auc_Ada = auc(fpr_Ada, tpr_Ada)  
        print(roc_auc_Ada)
        
        ###grid search for Ada
        #param_test1 = { 'min_samples_split':[10],'min_samples_leaf':[10]}
        #gsearch1 = GridSearchCV(estimator = RandomForestClassifier(n_estimators = 70,
                        #max_features='sqrt' ,random_state=10,class_weight = 'balanced',max_depth =22), 
                       #param_grid = param_test1, scoring='roc_auc',cv=5)
        #gsearch1.fit(x_train, y_train)
        #print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)

        print('Start NN')
        model_NN = MLPClassifier(solver='adam', alpha=1e-5,
                    hidden_layer_sizes=(80,20), random_state=1,
                    learning_rate_init = 0.01, batch_size = 'auto')
        x_train_NN = x_train
        y_train_NN = y_train
        x_test_NN = x_test
        model_NN.fit(x_train_NN,y_train_NN)
        pre_NN = model_NN.predict_proba(x_test_NN)
        fpr_NN, tpr_NN, thresholds_NN = roc_curve(y_test, pre_NN[:,1])  
        roc_auc_NN = auc(fpr_NN, tpr_NN)  
        print(roc_auc_NN)

        ###grid search for MLP
        #find alpha
        '''print('find alpha')
        param_test1 = { 'alpha':[1e-5,1e-4,0.001,0.01,0.1,10.0,100.0,1000.0]}
        gsearch1 = GridSearchCV(estimator = MLPClassifier(solver='adam',
                    hidden_layer_sizes=(5, 2), random_state=1,
                    learning_rate_init = 0.001, batch_size = 'auto'), 
                       param_grid = param_test1, scoring='roc_auc',cv=5)
        gsearch1.fit(x_train, y_train)
        print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
        '''

        #find learning rate  
        '''print('find learning rate')
        param_test2 = { 'learning_rate_init':[1e-5,1e-4,0.001,0.01,0.1,10.0,100.0,1000.0]}
        gsearch2 = GridSearchCV(estimator = MLPClassifier(solver='adam', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1,batch_size = 'auto'), 
                       param_grid = param_test2, scoring='roc_auc',cv=5)
        gsearch2.fit(x_train, y_train)
        print(gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)
        '''

        #find hidden layer
        '''print('find hidden_layer_sizes_node')
        param_test3 = { 'hidden_layer_sizes':[(30,),(35,),(40,),(50,),(60,),(68,),(75,),(80,),(90,),(100,)]}
        gsearch3 = GridSearchCV(estimator = MLPClassifier(solver='adam', alpha=1e-5,learning_rate_init = 0.01,
                    random_state=1,batch_size = 'auto'), 
                       param_grid = param_test3, scoring='roc_auc',cv=5)
        gsearch3.fit(x_train, y_train)
        print(gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_)
        '''

        #find hidden layer  
        '''print('find hidden_layer_sizes')
        param_test3 = { 'hidden_layer_sizes':[(80,20,20,10,15,5),(80,20,20,10,15,10),(80,20,20,10,15,15),(80,20,20,10,15,20),(80,20,20,10,15,25),(80,20,20,10,15,30)]}
        gsearch3 = GridSearchCV(estimator = MLPClassifier(solver='adam', alpha=1e-5,learning_rate_init = 0.01,
                    random_state=1,batch_size = 'auto'), 
                       param_grid = param_test3, scoring='roc_auc',cv=5)
        gsearch3.fit(x_train, y_train)
        print(gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_)
        '''
         #find hidden layer  
        '''print('find hidden_layer_sizes')
        param_test3 = { 'hidden_layer_sizes':[(80),(80,20),(80,20,20),(80,20,20,10),(80,20,20,10,15),(80,20,20,10,15,10)]}
        gsearch3 = GridSearchCV(estimator = MLPClassifier(solver='adam', alpha=1e-5,learning_rate_init = 0.01,
                    random_state=1,batch_size = 'auto'), 
                       param_grid = param_test3, scoring='roc_auc',cv=5)
        gsearch3.fit(x_train, y_train)
        print(gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_)
        '''

        ###train model-------------------------------------------------------------------------------------------
        '''model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        pre = model.predict_proba(x_test)
        # confusion matrix
        confusion = confusion_matrix(y_test, predictions)
        print("\t" + "\t".join(str(x) for x in range(0, 2)))
        print("".join(["-"] * 50))
        for ii in range(0, 2):
            jj = ii
            print("%i:\t" % jj + "\t".join(str(confusion[ii][x]) for x in range(0, 2)))

        print(pre)
        print(f1_score(y_test, predictions))
        print(precision_score(y_test, predictions))
        print(recall_score(y_test, predictions))
        print(roc_auc_score(y_test, predictions))
        # print(classification_report(y_test, predictions))
        '''
    print('Finish!')

    print('Start blend')

    w = [0.0,0.0,0.6,0.3,0.1,0.0]
    #pre_blend = w[0]*pre_LR[:,1]+w[1]*pre_SVM+w[2]*pre_RF[:,1]+w[3]*pre_GBM+w[4]*pre_Ada[:,1]+w[5]*pre_FM[:,1]+w[6]*pre_RF_bag+w[7]*pre_Xgb
    pre_blend = w[0]*pre_LR[:,1]+w[1]*pre_SVM[:,1]+w[2]*pre_RF[:,1]+w[3]*pre_GBM+w[4]*pre_Ada[:,1]+w[5]*pre_NN[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, pre_blend)  
    roc_auc = auc(fpr, tpr)  
    print(roc_auc)

if __name__ == "__main__":

    train_ui_up_mp_um_double_pca_sim_slope_fillna_features = features()
    classification(train_ui_up_mp_um_double_pca_sim_slope_fillna_features)