# extract trend features

import pandas as pd
import numpy as np
import psycopg2
import traceback
from scipy.stats import linregress

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

output_path = "C:/Users/yawen/Google Drive/spatial temporal data analytics/urban computing/PurchasingBehavior/feature_analysis_output/"

"""
get up, mp, um features
"""
cur.execute("select * from train_ui_up_mp_um_profile")
train_ui_up_mp_um_features = pd.DataFrame(cur.fetchall(), columns = [i[0] for i in cur.description])

def up_click_slope(row):
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [row['up_count_click_may'], row['up_count_click_jun'], row['up_count_click_jul'], row['up_count_click_aug'], row['up_count_click_sep'], row['up_count_click_oct'], row['up_count_click_nov']]
    return linregress(x, y)[0]

def up_cart_slope(row):
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [row['up_count_cart_may'], row['up_count_cart_jun'], row['up_count_cart_jul'], row['up_count_cart_aug'], row['up_count_cart_sep'], row['up_count_cart_oct'], row['up_count_cart_nov']]
    return linregress(x, y)[0]

def up_purchase_slope(row):
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [row['up_count_purchase_may'], row['up_count_purchase_jun'], row['up_count_purchase_jul'], row['up_count_purchase_aug'], row['up_count_purchase_sep'], row['up_count_purchase_oct'], row['up_count_purchase_nov']]
    return linregress(x, y)[0]

def up_favorite_slope(row):
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [row['up_count_favorite_may'], row['up_count_favorite_jun'], row['up_count_favorite_jul'], row['up_count_favorite_aug'], row['up_count_favorite_sep'], row['up_count_favorite_oct'], row['up_count_favorite_nov']]
    return linregress(x, y)[0]

# user, slope
train_ui_up_mp_um_features['up_click_slope'] = train_ui_up_mp_um_features.apply(up_click_slope, axis = 1)
train_ui_up_mp_um_features['up_cart_slope'] = train_ui_up_mp_um_features.apply(up_cart_slope, axis = 1)
train_ui_up_mp_um_features['up_purchase_slope'] = train_ui_up_mp_um_features.apply(up_purchase_slope, axis = 1)
train_ui_up_mp_um_features['up_favorite_slope'] = train_ui_up_mp_um_features.apply(up_favorite_slope, axis = 1)

def mp_click_slope(row):
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [row['mp_count_click_may'], row['mp_count_click_jun'], row['mp_count_click_jul'], row['mp_count_click_aug'], row['mp_count_click_sep'], row['mp_count_click_oct'], row['mp_count_click_nov']]
    return linregress(x, y)[0]

def mp_cart_slope(row):
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [row['mp_count_cart_may'], row['mp_count_cart_jun'], row['mp_count_cart_jul'], row['mp_count_cart_aug'], row['mp_count_cart_sep'], row['mp_count_cart_oct'], row['mp_count_cart_nov']]
    return linregress(x, y)[0]

def mp_purchase_slope(row):
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [row['mp_count_purchase_may'], row['mp_count_purchase_jun'], row['mp_count_purchase_jul'], row['mp_count_purchase_aug'], row['mp_count_purchase_sep'], row['mp_count_purchase_oct'], row['mp_count_purchase_nov']]
    return linregress(x, y)[0]

def mp_favorite_slope(row):
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [row['mp_count_favorite_may'], row['mp_count_favorite_jun'], row['mp_count_favorite_jul'], row['mp_count_favorite_aug'], row['mp_count_favorite_sep'], row['mp_count_favorite_oct'], row['mp_count_favorite_nov']]
    return linregress(x, y)[0]

# merchant, slope
train_ui_up_mp_um_features['mp_click_slope'] = train_ui_up_mp_um_features.apply(mp_click_slope, axis = 1)
train_ui_up_mp_um_features['mp_cart_slope'] = train_ui_up_mp_um_features.apply(mp_cart_slope, axis = 1)
train_ui_up_mp_um_features['mp_purchase_slope'] = train_ui_up_mp_um_features.apply(mp_purchase_slope, axis = 1)
train_ui_up_mp_um_features['mp_favorite_slope'] = train_ui_up_mp_um_features.apply(mp_favorite_slope, axis = 1)

def um_click_slope(row):
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [row['um_count_click_may'], row['um_count_click_jun'], row['um_count_click_jul'], row['um_count_click_aug'], row['um_count_click_sep'], row['um_count_click_oct'], row['um_count_click_nov']]
    return linregress(x, y)[0]

def um_cart_slope(row):
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [row['um_count_cart_may'], row['um_count_cart_jun'], row['um_count_cart_jul'], row['um_count_cart_aug'], row['um_count_cart_sep'], row['um_count_cart_oct'], row['um_count_cart_nov']]
    return linregress(x, y)[0]

def um_purchase_slope(row):
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [row['um_count_purchase_may'], row['um_count_purchase_jun'], row['um_count_purchase_jul'], row['um_count_purchase_aug'], row['um_count_purchase_sep'], row['um_count_purchase_oct'], row['um_count_purchase_nov']]
    return linregress(x, y)[0]

def um_favorite_slope(row):
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [row['um_count_favorite_may'], row['um_count_favorite_jun'], row['um_count_favorite_jul'], row['um_count_favorite_aug'], row['um_count_favorite_sep'], row['um_count_favorite_oct'], row['um_count_favorite_nov']]
    return linregress(x, y)[0]

# user-merchant, slope
train_ui_up_mp_um_features['um_click_slope'] = train_ui_up_mp_um_features.apply(um_click_slope, axis = 1)
train_ui_up_mp_um_features['um_cart_slope'] = train_ui_up_mp_um_features.apply(um_cart_slope, axis = 1)
train_ui_up_mp_um_features['um_purchase_slope'] = train_ui_up_mp_um_features.apply(um_purchase_slope, axis = 1)
train_ui_up_mp_um_features['um_favorite_slope'] = train_ui_up_mp_um_features.apply(um_favorite_slope, axis = 1)

# export to csv
train_trend_features = train_ui_up_mp_um_features[['user_id', 'merchant_id', 'up_click_slope', 'up_cart_slope', 'up_purchase_slope', 'up_favorite_slope', 'mp_click_slope', 'mp_cart_slope', 'mp_purchase_slope', 'mp_favorite_slope', 'um_click_slope', 'um_cart_slope', 'um_purchase_slope', 'um_favorite_slope']]

train_trend_features.to_csv(output_path + 'train_trend_features.csv')
