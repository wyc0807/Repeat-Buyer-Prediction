import pandas as pd
import numpy as np
from numpy import array
import psycopg2
import math
import csv
import traceback
import matplotlib.pyplot as plt
from collections import defaultdict
import time

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn import cross_validation
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# connect to database and get cursor
try:
    conn = psycopg2.connect(database = 'complex features', user = 'postgres', host = 'localhost', port = '5432', password = '6371')

except psycopg2.Error as e:
    print("I am unable to connect to the database")
    print(e)
    print(e.pgcode)
    print(e.pgerror) 	
    print(traceback.format_exc())
cur = conn.cursor()
merchant_matrix=np.zeros((4995,4995),dtype=np.int)
cur.execute('select cast(um_purchase_num.user_id as int), cast(um_purchase_num.merchant_id as int) as merchant_id from um_purchase_num order by merchant_id')
um_purchase_num=pd.DataFrame(cur.fetchall(), columns = [i[0] for i in cur.description])


def extract_PCA_feature():
	t=um_purchase_num.filter(regex='mer')
	#print(t[0:100])
	
	for i in range(1,424171):
		temp=um_purchase_num[um_purchase_num['user_id']==i]
		for i in temp['merchant_id']:
			for j in temp['merchant_id']:
				merchant_matrix[i-1,j-1]+=1

	with open("merchant_matrix.csv","w") as f:
 		writer = csv.writer(f)
 		writer.writerows(merchant_matrix)
	#print(merchant_matrix[56,374],merchant_matrix[374,56])
	pca=PCA(n_components=10)
	prin_comp=pca.fit_transform(merchant_matrix)
	with open("pca.csv","w") as f:
 		writer = csv.writer(f)
 		writer.writerows(prin_comp)

if __name__ == '__main__':
	extract_PCA_feature()

	