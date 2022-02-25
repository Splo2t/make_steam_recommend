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
	csv_test = pd.read_csv('test.csv')
	print(csv_test.shape)
	print(csv_test.head())