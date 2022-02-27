'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''


import pandas as pd
import os


if __name__ == "__main__":
	csv_test = pd.read_csv('steam_reviews.csv')
	csv_test = csv_test.drop('review',axis=1)
	print(csv_test.shape)
	#csv_test = csv_test.groupby('app_id', sort=False).head(3000)
	print(csv_test.shape)
	print(csv_test.info)
	#my_data = csv_test.loc[0:50000]
	#my_data.to_csv('test_small.csv')
	csv_test.to_csv('drop_review_steam.csv')
 