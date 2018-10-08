import psycopg2
import csv
import pandas as pd
import numpy as np
from numpy import array
from pandas import DataFrame

import xgboost
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from xgboost import plot_importance
import matplotlib.pyplot as plt
import itertools



def evaluation(y_test,predictions,classifier_type):

    print("evaluation result for %s" % classifier_type)
   # confusion matrix
    confusion = confusion_matrix(y_test, predictions)
    print("\t" + "\t".join(str(x) for x in range(0, 2)))
    print("".join(["-"] * 50))
    for ii in range(0, 2):
        jj = ii
        print("%i:\t" % jj + "\t".join(str(confusion[ii][x]) for x in range(0, 2)))

    print 'F1 score'
    print(f1_score(y_test, predictions, average = "macro"))
    print 'Precision'
    print(precision_score(y_test, predictions, average = "macro"))
    print 'Recall'
    print(recall_score(y_test, predictions, average = "macro"))
    print 'Roc Auc'
    print(roc_auc_score(y_test, predictions, average = "macro")) 

    plt.figure()
    #plotConfusionMatrix(confusion,classifier_type)
    plt.show()

        
      
def plotConfusionMatrix(confusion_matrix, classifier):

    classes = ['not buy','buy']
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    fmt = 'd'
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

def data_processing():

    #import features
    all_features = pd.read_csv('train_ui_up_mp_um_double_pca_sim_slope_fillna_1212.csv')
    user_age_gener_feature =  pd.read_csv('user_age_gender_fillna_1212.csv')

    """
    complex_feature = pd.read_csv('complex_double11_fea.csv')
    basic_features = pd.read_csv('train_data_all.csv')
    user_profile_features = pd.read_csv('user_profile_features.csv')
    train_data_user_info_features = pd.read_csv('train_data_user_info_features.csv')
    train_data_user_features = pd.read_csv('train_data_user_features.csv')
    train_data_user_merchant_profile_features = pd.read_csv('train_data_user_merchant_profile_features.csv')
    train_data_merchant_features = pd.read_csv('train_data_merchant_features.csv')
    """

    print 'import data csv successful' 

    #basic features
    all_x = all_features

    ui_x = all_features.filter(items = ['age_range','gender','label'])
    ex_ui_x = all_features.drop(['age_range','gender'],axis = 1)

    ui_up_x = all_features.filter(regex= r'(up_count|label|age|gender)')
    ui_up = all_features.filter(regex = r'(up_count|age|gender)')
    ex_ui_up_x = all_features.drop(list(ui_up),axis=1)

    up_x = all_features.filter(regex = r'(up_count|label)')
    up = all_features.filter(regex = 'up_count')
    ex_up_x = all_features.drop(list(up),axis=1)

    mp_x = all_features.filter(regex = r'(mp_count|label)')
    mp = all_features.filter(regex = 'mp_count')
    ex_mp_x = all_features.drop(list(mp),axis=1)

    um_x = all_features.filter(regex = r'(um_count|label)')
    um = all_features.filter(regex = 'um_count')
    ex_um_x = all_features.drop(list(um),axis=1)

    #complex features
    double11_x = all_features.filter(regex = r'(double11|label)')
    double11 = all_features.filter(regex = 'double11')
    ex_double11_x = all_features.drop(list(double11),axis=1)

    slope_x = all_features.filter(regex = r'(slope|label)')
    slope = all_features.filter(regex = 'slope')
    ex_slope_x = all_features.drop(list(slope),axis=1)

    pca_x = all_features.filter(regex = r'(pca|label)')
    pca = all_features.filter(regex = 'pca')
    ex_pca_x = all_features.drop(list(pca),axis=1)

    sim_x = all_features.filter(regex = r'(sim|label)')
    sim = all_features.filter(regex = 'sim')
    ex_sim_x = all_features.drop(list(sim),axis=1)


    feature_groups = [all_x, ui_x, ex_ui_x, up_x, ex_up_x, mp_x, ex_mp_x, um_x, ex_um_x, double11_x, ex_double11_x,slope_x, ex_slope_x, pca_x, ex_pca_x,sim_x,ex_sim_x,ui_up_x,ex_ui_up_x]

    x = feature_groups[17]

    y = all_features['label']

    #desampling: to make the labeled data not to sparse
    repeat_index = x[y == 1].index
    not_repeat_index = x[y == 0].index

    repeat_index_desample = np.random.choice(repeat_index, int(len(x)*0.05))
    not_repeat_index_desample = np.random.choice(not_repeat_index, int(len(x)*0.20))

    print len(repeat_index_desample)
    print len(not_repeat_index_desample)

    #combine the two array of indexes and sort in order
    index = np.concatenate((repeat_index_desample, not_repeat_index_desample))
    index = np.sort(index)

    x = x.loc[index]

    #randomly select 20% data as test data for modeling
    #for feature importance, no need to slice data
    x_test = x.sample(frac=0)
    y_test = 0 #x_test['label']
    #y_test = y.loc[x_test.index]
    x_test = x_test.drop('label',axis=1)

    x_train = x.loc[~x.index.isin(x_test.index)]
    y_train =x_train['label']
    x_train = x_train.drop('label',axis=1)

    #normalize the data
    #for feature importance analyze, no need to normalize training data
    #x_train = x_train.apply(lambda x_train: (x_train - x_train.min()) / (x_train.max() - x_train.min()))

    return x_train, y_train, x_test, y_test


def feature_performance_xgboost(x_train,y_train,x_test,y_test):

    #brute force scan for all parameters, here are the tricks
    #usually max_depth is 6,7,8
    #learning rate is around 0.05, but small changes may make big diff
    #tuning min_child_weight subsample colsample_bytree can have 
    #much fun of fighting against overfit 
    #n_estimators is how many round of boosting
    #finally, ensemble xgboost with multiple seeds may reduce variance
    parameters = {'nthread':[3], #when use hyperthread, xgboost may become slower
                'gamma':[0.001],
                'max_delta_step':[5],
                'objective':['binary:logistic'],
                'learning_rate':[0.04], #so called `eta` value
                'max_depth':[7],
                'min_child_weight':[200],
                'silent':[1],
                'subsample':[0.8],
                'colsample_bytree':[0.7],
                'n_estimators':[10], #number of trees, change it to 1000 for better results
                'missing':[-999],
                'seed':[1337]}

    xgboost_model = xgboost.XGBClassifier(n_estimators=100,nthread=3,gamma=0.001,max_delta_step=5,
                                        learning_rate=0.04,max_depth=7,min_child_weight=200,
                                        subsample=0.8,colsample_bytree=0.7)

    sector = xgboost_model.fit(x_train,y_train,eval_metric='auc',verbose=False)

    plot_importance(xgboost_model)
    plt.show()

    #plt.bar(range(len(xgboost_model.feature_importances_)), xgboost_model.feature_importances_)
    #plt.show()

#    use grid search to fine optimal parameters
#    xgboost_model = xgboost.XGBClassifier()
#    clf = GridSearchCV(xgboost_model, parameters, n_jobs=5, 
#                    cv=5, scoring='roc_auc', verbose=2, refit=True)

#    clf.fit(x_train,y_train)

#    print clf.best_estimator_
#    print clf.best_params_
#    print clf.best_score_

#evaluation(y_test, lr_predictions,'XGBoost')

def plot_barchart():
    #plot bar chart
    # data to plot
    n_groups = 7
    auc_score = (0.544, 0.559, 0.607, 0.610,0.593,0.622,0.607)
    ex_auc_score = (0.658, 0.656, 0.652, 0.651,0.660,0.657,0.658)

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.25
    opacity = 0.8

    rects1 = plt.bar(index, auc_score, bar_width,
                    alpha=opacity,
                    color='b',
                    label='AUC')

    rects2 = plt.bar(index + bar_width, ex_auc_score, bar_width,
                    alpha=opacity,
                    color='g',
                    label='Exclude AUC')

    plt.xlabel('Feature groups')
    plt.ylabel('AUC scores')
    plt.title('Feature Performance')
    plt.xticks(index + bar_width, ('ui', 'up', 'mp', 'um','11.11','slope','pca'))
    plt.legend()

    plt.tight_layout()
    plt.show()
    
#plotBarchart()  

def Linear_regression(x_train,y_train,x_test,y_test):

    lr_model = LogisticRegression(class_weight='balanced')
    lr_model.fit(x_train,y_train)
    lr_predictions = lr_model.predict(x_test)

    """
    plt.scatter(y_train, y_test,  color='black')
    plt.plot(x_test, lr_predictions, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()
    """

    evaluation(y_test, lr_predictions,'LogisticRegression')

def Random_forest(x_train,y_train,x_test,y_test):

    #Random Forest classifer
    rf_model=RandomForestClassifier(n_estimators=100, criterion='gini')
    #param_grid = {
    #    'n_estimators': [50, 100, 200],
    #    'criterion': ['entropy', 'gini']
    #}
    rf_model.fit(x_train,y_train)
    rf_predictions = rf_model.predict(x_test)

    evaluation(y_test, rf_predictions,'RandomForest')


if __name__ == "__main__":

    x_train, y_train, x_test, y_test = data_processing()
    #Linear_regression(x_train, y_train, x_test, y_test)
    #Random_forest(x_train, y_train, x_test, y_test)
    feature_performance_xgboost(x_train, y_train,x_test, y_test)
    #plot_barchart()

    print('Finish!')

