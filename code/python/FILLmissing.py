# fill the missing data in gender and age_group

import pandas as pd
import numpy as np
from numpy import array
import psycopg2
import math
import traceback
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, classification_report

import seaborn as sns

# output path
input_path = "C:/Users/yawen/Google Drive/spatial temporal data analytics/urban computing/PurchasingBehavior/feature_analysis_output/"
output_path = "C:/Users/yawen/Google Drive/spatial temporal data analytics/urban computing/PurchasingBehavior/feature_analysis_output/"

# connect to database and get cursor
try:
    conn = psycopg2.connect(database = 'tmall', user = 'postgres', host = 'localhost', port = '5432', password = '1234')

except psycopg2.Error as e:
    print("I am unable to connect to the database")
    print(e)
    print(e.pgcode)
    print(e.pgerror)
    print(traceback.format_exc())

cur = conn.cursor()

def purchase_group(row):

    if 0 <= row['up_count_purchase'] <= 10:
        return '1-10'
    if 11 <= row['up_count_purchase'] <= 20:
        return '11-20'
    if 21 <= row['up_count_purchase'] <= 30:
        return '21-30'
    if 31 <= row['up_count_purchase'] <= 40:
        return '31-40'
    if 41 <= row['up_count_purchase'] <= 50:
        return '41-50'
    if 51 <= row['up_count_purchase'] <= 60:
        return '51-60'
    if 61 <= row['up_count_purchase'] <= 70:
        return '61-70'
    if 71 <= row['up_count_purchase'] <= 80:
        return '71-80'
    if 81 <= row['up_count_purchase'] <= 90:
        return '81-90'
    if 91 <= row['up_count_purchase'] <= 100:
        return '91-100'
    if 101 <= row['up_count_purchase'] <= 110:
        return '101-110'
    if 111 <= row['up_count_purchase'] <= 120:
        return '111-120'
    if 121 <= row['up_count_purchase'] <= 130:
        return '121-130'
    if 131 <= row['up_count_purchase'] <= 140:
        return '131-140'
    if 141 <= row['up_count_purchase'] <= 150:
        return '141-150'
    if row['up_count_purchase'] > 150:
        return '>150'

def mode(a):
    u, c = np.unique(a, return_counts=True)
    return u[c.argmax()]

def fill_missing():
    """
    get user features
    """
    cur.execute("select user_id, age_range, gender, up_count_purchase from train_ui_up_mp_um_profile")
    user_features = pd.DataFrame(cur.fetchall(), columns = [i[0] for i in cur.description])
    user_features = user_features.drop_duplicates(keep = 'first')

    # form purchase group
    user_features['purchase_group'] = user_features.apply(lambda row: purchase_group(row), axis = 1)

    # change 0.0 to NaN
    user_features['age_range'] = user_features['age_range'].replace(0.0, np.NaN)
    # change 2.0 to NaN
    user_features['gender'] = user_features['gender'].replace(2.0, np.NaN)

    user_features = user_features.fillna(user_features.groupby('purchase_group').transform('median'))
    user_age_gender_fillna = user_features[['user_id', 'age_range', 'gender']]
    user_age_gender_fillna.to_csv(output_path + 'user_age_gender_fillna_1212.csv')

if __name__ == "__main__":

    fill_missing()
